# ruff: noqa: N802, N815
# FakeACL mirrors the public AscendCL API names.
"""The board executor saves evidence; precision decisions belong to the host."""

from contextlib import redirect_stderr, redirect_stdout
import ctypes as ct
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import runner


def half(values):
    return struct.pack("<" + "e" * len(values), *values)


class FakeACL:
    """Host allocations stand in for device memory; dataset order is deliberately reversed."""

    def __init__(self, mode="normal", mismatch=None, dtype="fp16"):
        self.mode, self.mismatch, self.dtype = mode, mismatch, dtype
        self.memory, self.buffers, self.datasets = {}, {}, {}
        self.events, self.serial, self.executions = [], 100, 0
        self.copies = []

    def aclInit(self, _):
        self.events.append("init")
        return 0

    def aclFinalize(self):
        self.events.append("finalize")
        return 0

    def aclrtSetDevice(self, device):
        self.events.append(("device", device))
        return 0

    def aclrtResetDevice(self, device):
        self.events.append(("reset", device))
        return 0

    def aclmdlLoadFromFile(self, _, model):
        if not any(isinstance(e, tuple) and e[0] == "device" for e in self.events):
            raise AssertionError("model loaded before selecting device")
        model._obj.value = 7
        self.events.append("load")
        return 0

    def aclmdlUnload(self, _):
        self.events.append("unload")
        return 0

    def aclmdlCreateDesc(self):
        return 3

    def aclmdlDestroyDesc(self, _):
        self.events.append("destroy_desc")
        return 0

    def aclmdlGetDesc(self, *_):
        return 0

    def aclmdlGetNumInputs(self, _):
        return 3 if self.mismatch == "count" else 2

    def aclmdlGetNumOutputs(self, _):
        return 1

    def aclmdlGetInputIndexByName(self, _, name, index):
        index._obj.value = 0 if self.mismatch == "bijection" else {b"y": 0, b"x": 1}[name]
        return 0

    def aclmdlGetOutputNameByIndex(self, *_):
        return b"other" if self.mismatch == "output_name" else b"z"

    def aclmdlGetInputDims(self, _, index, dimensions):
        dimensions._obj.dimCount = 1
        dimensions._obj.dims[0] = 3 if self.mismatch == "shape" else 2
        return 0

    aclmdlGetOutputDims = aclmdlGetInputDims

    def aclmdlGetInputDataType(self, *_):
        return 0 if self.mismatch == "dtype" or self.dtype == "fp32" else 1

    aclmdlGetOutputDataType = aclmdlGetInputDataType

    def aclmdlGetInputSizeByIndex(self, *_):
        return 8 if self.mismatch == "bytes" or self.dtype == "fp32" else 4

    aclmdlGetOutputSizeByIndex = aclmdlGetInputSizeByIndex

    def aclrtMalloc(self, pointer, size, _):
        memory = ct.create_string_buffer(size)
        address = ct.addressof(memory)
        self.memory[address] = memory
        pointer._obj.value = address
        return 0

    def aclrtFree(self, pointer):
        del self.memory[pointer.value]
        return 0

    def aclrtMemcpy(self, destination, _, source, size, direction):
        self.copies.append((self.executions, direction, size))
        ct.memmove(destination, source, size)
        return 0

    def aclCreateDataBuffer(self, address, size):
        self.serial += 1
        self.buffers[self.serial] = (address, size)
        return self.serial

    def aclDestroyDataBuffer(self, buffer):
        del self.buffers[buffer]
        return 0

    def aclmdlCreateDataset(self):
        self.serial += 1
        self.datasets[self.serial] = []
        return self.serial

    def aclmdlDestroyDataset(self, dataset):
        del self.datasets[dataset]
        return 0

    def aclmdlAddDatasetBuffer(self, dataset, buffer):
        self.datasets[dataset].append(buffer)
        return 0

    def aclmdlExecute(self, _, inputs, outputs):
        self.executions += 1
        if self.mode == "failure" or (self.mode == "failure_second" and self.executions == 2):
            return 42
        decoded = []
        format_ = "<ff" if self.dtype == "fp32" else "<ee"
        for buffer in self.datasets[inputs]:
            address, size = self.buffers[buffer]
            decoded.append(struct.unpack(format_, ct.string_at(address, size)))
        y, x = decoded
        payload = struct.pack(format_, *[10 * a + b for a, b in zip(x, y)])
        if self.mode == "nan":
            payload = half([float("nan"), 24])
        if self.mode == "partial":
            payload = payload[:2]
        if self.mode == "later_output" and self.executions > 2:
            payload = half([self.executions, 24])
        address, _ = self.buffers[self.datasets[outputs][0]]
        ct.memmove(address, payload, len(payload))
        return 0


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request_file, self.om = self.root / "execution.json", self.root / "model.om"
        self.om.write_bytes(b"test-model-only")
        self.request = {
            "schema_version": 1,
            "kind": "pypto_pro_execution_request",
            "runtime_dependencies": "device_kernel_only",
            "om": {"file": "model.om", "sha256": runner.sha256(self.om.read_bytes())},
            "inputs": [],
            "output": {"name": "z", "model_names": ["z"]},
        }
        for name, values in (("x", [1, 2]), ("y", [3, 4])):
            raw = half(values)
            (self.root / (name + ".bin")).write_bytes(raw)
            self.request["inputs"].append({"name": name, "file": name + ".bin", "sha256": runner.sha256(raw)})
        self.write_request()

    def write_request(self):
        self.request_file.write_text(json.dumps(self.request))

    def validated(self):
        return runner.load_execution(self.request_file)

    def invoke(self, acl, suffix="direct", **profile):
        destination = self.root / suffix
        destination.mkdir()
        return runner.execute(acl, self.validated(), destination, **profile), destination

    def assert_clean(self, acl):
        self.assertEqual((acl.memory, acl.buffers, acl.datasets), ({}, {}, {}))
        self.assertEqual(acl.events[-2:], [("reset", 0), "finalize"])
        self.assertIn("unload", acl.events)

    def cli(self, acl, suffix="cli", extra=None):
        directory = self.root / suffix
        arguments = ["--execution", str(self.request_file), "--output", str(directory)]
        arguments.extend(extra or [])
        if acl is None:
            with redirect_stdout(io.StringIO()):
                code = runner.main(arguments)
        else:
            with patch.object(runner, "load_acl", return_value=acl), redirect_stdout(io.StringIO()):
                code = runner.main(arguments)
        return code, directory, json.loads((directory / "execution.json").read_text())

    def test_runtime_mapping_descriptors_and_raw_rounds_without_golden(self):
        acl = FakeACL()
        report, directory = self.invoke(acl)
        self.assertEqual(report["execution_calls"], 2)
        self.assertEqual([x["name"] for x in report["inputs"]], ["y", "x"])
        self.assertEqual(report["output"]["shape"], [2])
        self.assertEqual(report["output"]["acl_dtype"], 1)
        self.assertEqual(report["output"]["bytes"], 4)
        self.assertEqual(report["output"]["model_name"], "z")
        self.assertEqual([x["sentinel"] for x in report["rounds"]], [127, 128])
        for item in report["rounds"]:
            raw = (directory / item["file"]).read_bytes()
            self.assertEqual(raw, half([13, 24]))
            self.assertEqual(runner.sha256(raw), item["sha256"])
        self.assertFalse((self.root / "z.bin").exists())
        self.assert_clean(acl)

    def test_model_abi_failures_release_resources(self):
        for mismatch in ("count", "bijection", "bytes", "output_name"):
            with self.subTest(mismatch=mismatch):
                acl = FakeACL(mismatch=mismatch)
                with self.assertRaises(ValueError):
                    self.invoke(acl, suffix=mismatch)
                self.assertEqual(acl.executions, 0)
                self.assert_clean(acl)

    def test_inconsistent_runtime_shape_dtype_and_size_are_rejected(self):
        for mismatch in ("shape", "dtype"):
            with self.subTest(mismatch=mismatch):
                acl = FakeACL(mismatch=mismatch)
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    self.invoke(acl, suffix=mismatch)
                self.assertEqual(acl.executions, 0)
                self.assert_clean(acl)

    def test_fp32_inputs_outputs_and_small_chunks_preserve_raw_bytes(self):
        for name, values in (("x", [1, 2]), ("y", [3, 4])):
            raw = struct.pack("<ff", *values)
            (self.root / (name + ".bin")).write_bytes(raw)
            next(item for item in self.request["inputs"] if item["name"] == name)["sha256"] = runner.sha256(raw)
        self.write_request()
        acl = FakeACL(dtype="fp32")
        with patch.object(runner, "IO_CHUNK_BYTES", 3):
            report, directory = self.invoke(acl, warmup=1, repeat=2)
        self.assertEqual(report["output"]["dtype"], "fp32")
        self.assertEqual(report["output"]["bytes"], 8)
        for name in ("actual.bin", "actual_second.bin", "profile_actual.bin"):
            self.assertEqual((directory / name).read_bytes(), struct.pack("<ff", 13, 24))
        self.assertTrue(all(count <= 3 for _, _, count in acl.copies))
        self.assertEqual({call for call, direction, _ in acl.copies if direction == 2}, {1, 2, 5})
        self.assert_clean(acl)

    def test_loading_inputs_keeps_paths_and_streams_payload_hashes(self):
        read_bytes = Path.read_bytes

        def no_tensor_read(path):
            if path.suffix in (".bin", ".om"):
                raise AssertionError("Payload must not be materialized as bytes")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", no_tensor_read), patch.object(runner, "IO_CHUNK_BYTES", 3):
            request = self.validated()
            self.assertTrue(all("_data" not in item and item["_bytes"] == 4 for item in request["inputs"]))
            directory = self.root / "streamed"
            directory.mkdir()
            runner.execute(FakeACL(), request, directory)
        self.assertEqual((directory / "actual.bin").read_bytes(), half([13, 24]))

    def test_input_changed_after_validation_is_rejected_and_resources_freed(self):
        request = self.validated()
        (self.root / "y.bin").write_bytes(half([8, 9]))
        acl = FakeACL()
        with self.assertRaisesRegex(ValueError, "changed during H2D"):
            runner.execute(acl, request, self.root)
        self.assertEqual(acl.executions, 0)
        self.assert_clean(acl)

    def test_partial_d2h_failure_retains_file_without_claiming_a_completed_round(self):
        acl = FakeACL()
        copy = acl.aclrtMemcpy
        second_round_chunks = []

        def fail_second_chunk(destination, capacity, source, count, direction):
            if direction == 2 and acl.executions == 2:
                second_round_chunks.append(count)
                if len(second_round_chunks) == 2:
                    return 77
            return copy(destination, capacity, source, count, direction)

        acl.aclrtMemcpy = fail_second_chunk
        with patch.object(runner, "IO_CHUNK_BYTES", 2):
            code, directory, report = self.cli(acl, suffix="partial-d2h")
        self.assertEqual(code, 1)
        self.assertEqual(report["execution_calls"], 2)
        self.assertEqual(len(report["rounds"]), 1)
        self.assertEqual((directory / "actual.bin").read_bytes(), half([13, 24]))
        self.assertEqual((directory / "actual_second.bin").stat().st_size, 2)
        self.assertIn("77", report["error"])
        self.assert_clean(acl)

    def test_large_sentinel_fill_uses_bounded_chunks_and_64bit_offsets(self):
        size, base, calls = 1061683200, 0x100000000, []

        class FillACL:
            def aclrtMemcpy(self, destination, capacity, source, count, direction):
                self.assertions(destination, capacity, source, count, direction)
                return 0

            def assertions(self, destination, capacity, source, count, direction):
                calls.append((destination.value, count))
                if count > runner.IO_CHUNK_BYTES or capacity != count or direction != 1:
                    raise AssertionError("Unbounded fill")
                if ct.string_at(source, 1) != b"\x80":
                    raise AssertionError("Wrong sentinel")

        runner._prefill_output(FillACL(), ct.c_void_p(base), size, 128)
        self.assertEqual(sum(count for _, count in calls), size)
        self.assertEqual(calls[0][0], base)
        self.assertEqual(calls[-1][0] + calls[-1][1], base + size)

    def test_unsupported_runtime_dtype_is_rejected_before_execution(self):
        acl = FakeACL()
        acl.aclmdlGetInputDataType = lambda *_: 3
        with self.assertRaisesRegex(ValueError, "FP16 or FP32"):
            self.invoke(acl, suffix="unsupported-dtype")
        self.assert_clean(acl)

    def test_om_and_input_hashes_are_checked_before_acl_load(self):
        self.om.write_bytes(b"another-model")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validated()
        self.request["om"]["sha256"] = runner.sha256(self.om.read_bytes())
        self.write_request()
        (self.root / "x.bin").write_bytes(half([9, 9]))
        with patch.object(runner, "load_acl") as load:
            code, _, report = self.cli(None)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            load.assert_not_called()

    def test_host_comparison_data_is_not_accepted(self):
        for name, value in (("golden", "absent.bin"), ("atol", 0), ("rtol", 0), ("binding", {}), ("case", {})):
            with self.subTest(name=name):
                self.write_request()
                data = dict(self.request)
                data[name] = value
                self.request_file.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "comparison data"):
                    self.validated()

    def test_schema_and_relative_payload_paths(self):
        for field, value in (("schema_version", 2), ("kind", "comparison"), ("runtime_dependencies", "host_plugin")):
            with self.subTest(field=field):
                data = dict(self.request)
                data[field] = value
                self.request_file.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "schema-1"):
                    self.validated()
        self.request["om"]["file"] = "../other.om"
        self.write_request()
        with self.assertRaisesRegex(ValueError, "relative"):
            self.validated()

    def test_duplicate_request_input_names_are_rejected(self):
        self.request["inputs"][1]["name"] = "x"
        self.write_request()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.validated()

    def test_cli_reports_execution_only(self):
        acl = FakeACL()
        code, directory, report = self.cli(acl)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "executed")
        self.assertEqual(report["kind"], "pypto_pro_execution_result")
        self.assertEqual(report["request_sha256"], runner.sha256(self.request_file.read_bytes()))
        self.assertEqual(report["om_sha256"], runner.sha256(self.om.read_bytes()))
        self.assertTrue((directory / "actual.bin").is_file())
        self.assertTrue((directory / "actual_second.bin").is_file())
        self.assertNotIn("passed", report)
        self.assertNotIn("mismatches", report)
        self.assertEqual(acl.executions, 2)
        profile = report["profiling_execution"]
        self.assertFalse(profile["enabled"])
        self.assertEqual(profile["status"], "not_requested")
        self.assertEqual(profile["total_execution_calls"], 2)
        self.assertEqual(profile["host_wall_seconds"], [])

    def test_profile_runs_raw_rounds_then_warmup_then_timing_without_more_copies(self):
        acl = FakeACL()
        with patch.object(runner.time, "perf_counter", side_effect=range(10)):
            report, directory = self.invoke(acl, warmup=2, repeat=3)
        profile = report["profiling_execution"]
        self.assertEqual(acl.executions, 7)
        self.assertEqual(report["execution_calls"], 2)
        self.assertEqual(profile["correctness_calls"], 2)
        self.assertEqual(profile["completed_warmup"], 2)
        self.assertEqual(profile["completed_repeat"], 3)
        self.assertEqual(profile["total_execution_calls"], 7)
        self.assertEqual(profile["attempted_execution_calls"], 7)
        self.assertEqual(profile["discard_first_executions"], 4)
        self.assertEqual(profile["status"], "completed")
        self.assertEqual(profile["host_wall_seconds"], [1, 1, 1])
        self.assertEqual(profile["timing_scope"], "synchronous_aclmdlExecute_host_wall")
        self.assertEqual(profile["count_scope"], "model_execution_not_individual_kernel")
        self.assertEqual([count for count, direction, _ in acl.copies if direction == 2], [1, 2, 7])
        self.assertTrue(all(count <= 2 or (count == 7 and direction == 2) for count, direction, _ in acl.copies))
        self.assertEqual((directory / "actual.bin").read_bytes(), half([13, 24]))
        self.assertEqual((directory / "actual_second.bin").read_bytes(), half([13, 24]))
        self.assertEqual(
            profile["final_output"],
            {
                "file": "profile_actual.bin",
                "bytes": 4,
                "sha256": runner.sha256((directory / "profile_actual.bin").read_bytes()),
            },
        )
        self.assert_clean(acl)

    def test_profile_final_output_is_read_after_repeats_without_comparing_on_board(self):
        report, directory = self.invoke(FakeACL(mode="later_output"), warmup=1, repeat=2)
        self.assertEqual((directory / "actual_second.bin").read_bytes(), half([13, 24]))
        self.assertEqual((directory / "profile_actual.bin").read_bytes(), half([5, 24]))
        self.assertEqual(report["profiling_execution"]["status"], "completed")
        self.assertNotIn("passed", report)

    def test_cli_profile_options_record_requested_and_completed_model_calls(self):
        code, _, report = self.cli(FakeACL(), extra=["--warmup", "1", "--repeat", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "executed")
        self.assertEqual(report["profiling_execution"]["warmup"], 1)
        self.assertEqual(report["profiling_execution"]["repeat"], 2)
        self.assertEqual(report["profiling_execution"]["completed_repeat"], 2)
        self.assertEqual(len(report["profiling_execution"]["host_wall_seconds"]), 2)

    def test_profile_repeat_without_warmup_starts_after_two_raw_rounds(self):
        report, _ = self.invoke(FakeACL(), repeat=1)
        profile = report["profiling_execution"]
        self.assertEqual(profile["completed_warmup"], 0)
        self.assertEqual(profile["completed_repeat"], 1)
        self.assertEqual(profile["discard_first_executions"], 2)
        self.assertEqual(profile["total_execution_calls"], 3)

    def test_invalid_profile_counts_stop_before_acl_load(self):
        for index, arguments in enumerate((["--warmup", "-1"], ["--repeat", "-1"], ["--warmup", "1"])):
            with self.subTest(arguments=arguments), patch.object(runner, "load_acl") as load:
                code, _, report = self.cli(None, suffix="bad-profile-" + str(index), extra=arguments)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["execution_calls"], 0)
                load.assert_not_called()

    def test_profile_failure_preserves_raw_rounds_and_actual_completed_counts(self):
        for failure_call, warmup, repeats in ((3, 2, 3), (5, 1, 3)):
            with self.subTest(failure_call=failure_call):
                acl = FakeACL()
                original = acl.aclmdlExecute

                def fail_at_call(*args):
                    if acl.executions + 1 == failure_call:
                        acl.executions += 1
                        return 42
                    return original(*args)

                acl.aclmdlExecute = fail_at_call
                code, directory, report = self.cli(
                    acl,
                    suffix="profile-failure-" + str(failure_call),
                    extra=["--warmup", str(warmup), "--repeat", str(repeats)],
                )
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["execution_calls"], 2)
                self.assertEqual(report["profiling_execution"]["status"], "failed")
                self.assertEqual(report["profiling_execution"]["total_execution_calls"], failure_call - 1)
                self.assertEqual(report["profiling_execution"]["attempted_execution_calls"], failure_call)
                self.assertEqual(report["profiling_execution"]["completed_repeat"], max(0, failure_call - 3 - warmup))
                self.assertEqual((directory / "actual.bin").read_bytes(), half([13, 24]))
                self.assertEqual((directory / "actual_second.bin").read_bytes(), half([13, 24]))
                self.assert_clean(acl)

    def test_partial_outputs_are_saved_without_board_side_comparison(self):
        acl = FakeACL(mode="partial")
        code, directory, report = self.cli(acl)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "executed")
        self.assertNotEqual((directory / "actual.bin").read_bytes(), (directory / "actual_second.bin").read_bytes())
        self.assertEqual(len(report["rounds"]), 2)
        self.assertNotIn("dual_sentinel_mismatch", report)
        self.assert_clean(acl)

    def test_nonfinite_outputs_are_saved_without_interpretation(self):
        code, directory, report = self.cli(FakeACL(mode="nan"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "executed")
        self.assertEqual((directory / "actual.bin").read_bytes(), half([float("nan"), 24]))
        self.assertNotIn("nonfinite", report)

    def test_second_execution_failure_preserves_first_round_evidence(self):
        acl = FakeACL(mode="failure_second")
        code, directory, report = self.cli(acl)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["execution_calls"], 1)
        self.assertEqual(len(report["rounds"]), 1)
        self.assertEqual((directory / "actual.bin").read_bytes(), half([13, 24]))
        self.assertFalse((directory / "actual_second.bin").exists())
        self.assertIn("42", report["error"])
        self.assert_clean(acl)

    def test_existing_results_directory_is_not_reused(self):
        self.cli(FakeACL())
        with patch.object(runner, "load_acl") as load, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(["--execution", str(self.request_file), "--output", str(self.root / "cli")])
        self.assertEqual(code, 1)
        load.assert_not_called()

    def test_library_capability_missing(self):
        with patch.object(runner.ct, "CDLL", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "aclInit"):
                runner.load_acl()


if __name__ == "__main__":
    unittest.main()
