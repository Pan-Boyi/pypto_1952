"""Compiler-side acceptance and transfer boundaries; no SDK, SSH or device is used."""

import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_pro_atc.audit import audit_om_binary

from pypto_1952 import host, run_profile
from tests.binary_fixtures import device_elf, standard_om


def half(values):
    return struct.pack("<" + "e" * len(values), *values)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build, self.results = self.root / "build", self.root / "results"
        self.case_dir = self.build / "case"
        self.case_dir.mkdir(parents=True)
        self.om = self.build / "model.om"
        self.om.write_bytes(standard_om([(0, b"structural graph only"), (3, device_elf())]))
        records = []
        for name, values in (("x", [1, 2, 3, 4]), ("y", [5, 6, 7, 8]), ("z", [6, 8, 10, 12])):
            path = self.case_dir / (name + ".bin")
            path.write_bytes(half(values))
            records.append(
                {
                    "name": name,
                    "file": path.name,
                    "sha256": host.sha256(path),
                    "shape": [2, 2],
                    "dtype": "fp16",
                    "bytes": 8,
                }
            )
        self.case = {
            "schema_version": 1,
            "operator": "vector_add",
            "inputs": records[:2],
            "outputs": records[2:],
            "atol": 0,
            "rtol": 0,
        }
        self.case_path, self.binding_path = self.case_dir / "case.json", self.case_dir / "binding.json"
        self.case_path.write_text(json.dumps(self.case))
        self.binding = {
            "schema_version": 2,
            "case_sha256": host.sha256(self.case_path),
            "om_sha256": host.sha256(self.om),
            "runtime_dependencies": "device_kernel_only",
            "host_portability": audit_om_binary(self.om),
            "output_name_bindings": [
                {"logical_name": "z", "om_declared_names": ["z", "native_pro:0", "native_pro:0:z"]}
            ],
        }
        self.binding_path.write_text(json.dumps(self.binding))
        self.bundle = self.build / "bundle"
        run_profile._make_bundle(self.case_path, self.om, self.binding_path, self.bundle)
        self.request_path = self.bundle / "execution.json"
        self.request = json.loads(self.request_path.read_text())
        self.results.mkdir()
        golden = (self.case_dir / "z.bin").read_bytes()
        self.write_results(golden, golden)

    def write_results(self, first, second):
        report = {
            "schema_version": 1,
            "kind": "pypto_pro_execution_result",
            "status": "executed",
            "python_version": "3.7.9",
            "execution_calls": 2,
            "request_sha256": host.sha256(self.request_path),
            "om_sha256": host.sha256(self.om),
            "inputs": [
                dict(item, acl_dtype=host.DTYPES[item["dtype"]][2], model_index=index)
                for index, item in enumerate(self.case["inputs"])
            ],
            "output": dict(
                self.case["outputs"][0],
                acl_dtype=host.DTYPES[self.case["outputs"][0]["dtype"]][2],
                model_index=0,
                model_name="native_pro:0:z",
            ),
            "rounds": [],
        }
        for index, (name, raw) in enumerate((("actual.bin", first), ("actual_second.bin", second))):
            path = self.results / name
            path.write_bytes(raw)
            report["rounds"].append(
                {
                    "index": index + 1,
                    "file": name,
                    "sha256": host.sha256(path),
                    "bytes": len(raw),
                    "sentinel": (127, 128)[index],
                }
            )
        self.write_report(report)
        return report

    def write_report(self, report):
        (self.results / "execution.json").write_text(json.dumps(report))

    def report(self):
        return json.loads((self.results / "execution.json").read_text())

    def evaluate(self):
        return host.evaluate(self.build, self.results)

    def use_fp32(self):
        for item in self.case["inputs"] + self.case["outputs"]:
            path = self.case_dir / item["file"]
            values = [value for (value,) in struct.iter_unpack("<e", path.read_bytes())]
            path.write_bytes(struct.pack("<" + "f" * len(values), *values))
            item.update(dtype="fp32", bytes=4 * len(values), sha256=host.sha256(path))
        self.case_path.write_text(json.dumps(self.case))
        self.binding["case_sha256"] = host.sha256(self.case_path)
        self.binding_path.write_text(json.dumps(self.binding))
        shutil.rmtree(self.bundle)
        run_profile._make_bundle(self.case_path, self.om, self.binding_path, self.bundle)
        self.request = json.loads(self.request_path.read_text())
        raw = (self.case_dir / "z.bin").read_bytes()
        self.write_results(raw, raw)

    def test_bundle_contains_only_om_inputs_executor_and_request(self):
        expected = {"model.om", "runner.py", "execution.json", "README.txt", "x.bin", "y.bin"}
        inventory = json.loads((self.bundle / "bundle.json").read_text())
        self.assertEqual(inventory["schema_version"], 2)
        self.assertEqual(set(inventory["files"]), expected)
        self.assertEqual({path.name for path in self.bundle.iterdir()}, expected | {"bundle.json"})
        for name in ("z.bin", "case.json", "binding.json"):
            self.assertFalse((self.bundle / name).exists())
        self.assertEqual(
            set(self.request), {"schema_version", "kind", "runtime_dependencies", "om", "inputs", "output"}
        )
        self.assertEqual(set(self.request["inputs"][0]), {"name", "file", "sha256"})
        self.assertEqual(set(self.request["output"]), {"name", "model_names"})
        host.validate_build(self.build)

    def test_exact_results_pass_only_on_compiler_host(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["precision_checked_on"], "compiler_host")
        self.assertTrue(result["dual_sentinel_consistent"])
        self.assertEqual([item["mismatches"] for item in result["comparisons"]], [0, 0])

    def test_fp32_acceptance_streams_tensors_without_read_bytes(self):
        self.use_fp32()
        read_bytes = Path.read_bytes

        def no_tensor_read(path):
            if path.suffix == ".bin":
                raise AssertionError("Numerical acceptance must stream tensor files")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", no_tensor_read), patch.object(host, "IO_CHUNK_BYTES", 7):
            result = self.evaluate()
        self.assertEqual(result["status"], "passed")
        self.assertEqual([item["elements"] for item in result["comparisons"]], [4, 4])
        report = self.report()
        report["output"]["acl_dtype"] = 1
        self.write_report(report)
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self.evaluate()

    def test_chunked_fp32_comparison_handles_tolerance_nonfinite_and_last_chunk(self):
        actual, golden = self.root / "actual.bin", self.root / "golden.bin"
        golden.write_bytes(struct.pack("<ffff", 1, 2, 3, 4))
        actual.write_bytes(struct.pack("<ffff", 1, 2, 3, 4.125))
        with patch.object(host, "IO_CHUNK_BYTES", 5):
            self.assertTrue(host.compare_files(actual, golden, atol=0.125, dtype="fp32")["passed"])
            report = host.compare_files(actual, golden, dtype="fp32")
            self.assertEqual(report["mismatches"], 1)
            self.assertEqual(report["max_absolute_error"], 0.125)
            self.assertFalse(host.files_equal(actual, golden))
            actual.write_bytes(struct.pack("<ffff", 1, 2, 3, float("inf")))
            report = host.compare_files(actual, golden, dtype="fp32")
            self.assertEqual(report["nonfinite"], 1)
            self.assertIsNone(report["max_absolute_error"])

    def test_comparison_rejects_bad_sizes_dtypes_and_tolerances(self):
        for actual, golden, options in (
            (b"123", b"123", {"dtype": "fp32"}),
            (b"12", b"1234", {}),
            (b"1234", b"1234", {"dtype": "int32"}),
            (b"1234", b"1234", {"atol": float("nan")}),
            (b"1234", b"1234", {"rtol": -1}),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                host.compare(actual, golden, **options)
        self.assertEqual(host.tensor_bytes({"shape": [1, 64, 2025, 2048], "dtype": "fp32"}), 1061683200)
        for record in ({"shape": [0, 2], "dtype": "fp32"}, {"shape": [2], "dtype": "int32"}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                host.tensor_bytes(record)

    def test_optional_numpy_and_stdlib_agree_for_both_float_dtypes(self):
        for dtype, format_ in (("fp16", "<eeee"), ("fp32", "<ffff")):
            for values in ([1.125, -2, 0, 8], [float("nan"), float("inf"), -0.0, 8]):
                with self.subTest(dtype=dtype, values=values):
                    actual = struct.pack(format_, *values)
                    golden = struct.pack(format_, 1, -2, 0, 8)
                    with patch.dict(sys.modules, {"numpy": None}):
                        fallback = host.compare(actual, golden, atol=0.125, dtype=dtype)
                    self.assertEqual(host.compare(actual, golden, atol=0.125, dtype=dtype), fallback)
        extreme = struct.pack("<f", 3.4028234663852886e38)
        negative = struct.pack("<f", -3.4028234663852886e38)
        with patch.dict(sys.modules, {"numpy": None}):
            fallback = host.compare(extreme, negative, dtype="fp32")
        self.assertEqual(host.compare(extreme, negative, dtype="fp32"), fallback)
        self.assertTrue(fallback["max_absolute_error"] > 3.5e38)

    def test_signed_zero_rounds_fail_byte_identity_even_when_numerically_equal(self):
        self.use_fp32()
        first = struct.pack("<ffff", 0.0, 8, 10, 12)
        second = struct.pack("<ffff", -0.0, 8, 10, 12)
        (self.case_dir / "z.bin").write_bytes(first)
        self.case["outputs"][0]["sha256"] = host.sha256(self.case_dir / "z.bin")
        self.case_path.write_text(json.dumps(self.case))
        self.binding["case_sha256"] = host.sha256(self.case_path)
        self.binding_path.write_text(json.dumps(self.binding))
        self.write_results(first, second)
        result = self.evaluate()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["dual_sentinel_consistent"])
        self.assertTrue(all(item["passed"] for item in result["comparisons"]))

    def test_nan_outputs_fail_precision_on_host(self):
        raw = half([float("nan"), 8, 10, 12])
        self.write_results(raw, raw)
        result = self.evaluate()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["dual_sentinel_consistent"])
        self.assertEqual(result["comparisons"][0]["nonfinite"], 1)

    def test_different_rounds_fail_even_within_numeric_tolerance(self):
        self.case["atol"] = 1
        self.case_path.write_text(json.dumps(self.case))
        self.binding["case_sha256"] = host.sha256(self.case_path)
        self.binding_path.write_text(json.dumps(self.binding))
        self.write_results(half([6, 8, 10, 12]), half([6.5, 8, 10, 12]))
        result = self.evaluate()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["dual_sentinel_consistent"])
        self.assertTrue(all(item["passed"] for item in result["comparisons"]))

    def test_wrong_numerical_values_fail_even_when_rounds_match(self):
        self.write_results(half([6, 8, 10, 13]), half([6, 8, 10, 13]))
        result = self.evaluate()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["comparisons"][0]["mismatches"], 1)

    def test_wrong_runtime_shape_dtype_and_bytes_are_rejected(self):
        original = self.report()
        for direction, key, value in (
            ("output", "shape", [4]),
            ("output", "acl_dtype", 0),
            ("output", "bytes", 16),
            ("input", "shape", [1, 4]),
            ("input", "acl_dtype", 3),
            ("input", "sha256", "wrong"),
        ):
            with self.subTest(direction=direction, key=key):
                report = copy.deepcopy(original)
                (report["output"] if direction == "output" else report["inputs"][0])[key] = value
                self.write_report(report)
                with self.assertRaisesRegex(ValueError, "descriptor|contract"):
                    self.evaluate()

    def test_wrong_request_model_and_raw_hashes_are_rejected(self):
        original = self.report()
        for key in ("request_sha256", "om_sha256"):
            with self.subTest(key=key):
                self.write_report(dict(original, **{key: "another"}))
                with self.assertRaisesRegex(ValueError, "different request/OM"):
                    self.evaluate()
        self.write_report(original)
        (self.results / "actual_second.bin").write_bytes(half([0, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "Copied board output changed"):
            self.evaluate()

    def test_failed_execution_or_missing_round_is_not_precision_success(self):
        original = self.report()
        for changes in (
            {"status": "failed", "error": "device exception"},
            {"execution_calls": 1},
            {"rounds": original["rounds"][:1]},
        ):
            with self.subTest(changes=changes):
                self.write_report(dict(original, **changes))
                with self.assertRaisesRegex(ValueError, "both executions|two raw"):
                    self.evaluate()

    def test_old_bundle_and_changed_execution_request_are_rejected(self):
        inventory_path = self.bundle / "bundle.json"
        inventory = json.loads(inventory_path.read_text())
        inventory_path.write_text(json.dumps(dict(inventory, schema_version=1)))
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            host.validate_build(self.build)
        inventory_path.write_text(json.dumps(inventory))
        changed = dict(self.request, output={"name": "other", "model_names": ["other"]})
        self.request_path.write_text(json.dumps(changed))
        inventory["files"]["execution.json"] = host.sha256(self.request_path)
        inventory_path.write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, "request differs"):
            host.validate_build(self.build)

    def test_unlisted_golden_or_host_library_is_not_uploaded(self):
        for name in ("z.bin", "case.json", "libhost.so"):
            with self.subTest(name=name):
                path = self.bundle / name
                path.write_bytes(b"unlisted host-only artifact")
                try:
                    with self.assertRaisesRegex(ValueError, "bundle|Runtime"):
                        host.validate_build(self.build)
                finally:
                    path.unlink()

    def test_remote_run_uploads_only_bundle_and_compares_after_retrieval(self):
        output = self.root / "remote-output"
        output.mkdir()
        args = SimpleNamespace(
            build=self.build,
            host="board-alias",
            remote_dir="/tmp/pypto-test/session1",
            python="/usr/bin/python3.7",
            runtime_env="/opt/runtime/set_env.sh",
            device=0,
            timeout=90,
        )
        commands = []

        def fake_run(argv, **kwargs):
            self.assertEqual(kwargs["timeout"], 90)
            self.assertFalse(kwargs["check"])
            commands.append(argv)
            if argv[0] == "scp" and argv[2].startswith("board-alias:"):
                shutil.copytree(self.results, Path(argv[3]))
            return SimpleNamespace(returncode=0, stdout="stub transport only", stderr="")

        with patch.object(host.subprocess, "run", side_effect=fake_run):
            result = host.remote_run(args, output)
        self.assertEqual(result["status"], "passed")
        self.assertEqual([command[0] for command in commands], ["ssh", "scp", "ssh", "scp"])
        self.assertEqual(commands[1][2], str(self.bundle.resolve()))
        self.assertEqual(commands[1][3], "board-alias:/tmp/pypto-test/session1/bundle")
        self.assertIn("/usr/bin/python3.7 runner.py --execution execution.json", commands[2][2])
        self.assertIn("--output ../results", commands[2][2])
        self.assertNotIn("case.json", commands[2][2])
        self.assertNotIn("golden", commands[2][2])
        self.assertEqual(commands[3][2], "board-alias:/tmp/pypto-test/session1/results")
        self.assertTrue((output / "board/actual_second.bin").is_file())
        self.assertEqual(json.loads((output / "commands.json").read_text()), commands)

    def test_failed_board_execution_still_retrieves_evidence(self):
        output = self.root / "remote-failure"
        output.mkdir()
        args = SimpleNamespace(
            build=self.build,
            host="board-alias",
            remote_dir="/tmp/pypto-test/session2",
            python="python3.7",
            runtime_env=None,
            device=0,
            timeout=60,
        )
        calls = []

        def fake_run(argv, **_):
            calls.append(argv)
            if len(calls) == 4:
                shutil.copytree(self.results, Path(argv[3]))
            return SimpleNamespace(returncode=42 if len(calls) == 3 else 0, stdout="", stderr="device error")

        with patch.object(host.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "raw results and logs were retrieved"):
                host.remote_run(args, output)
        self.assertEqual(len(calls), 4)
        self.assertTrue((output / "board/execution.json").is_file())


if __name__ == "__main__":
    unittest.main()
