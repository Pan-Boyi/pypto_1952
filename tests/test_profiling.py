"""Profiling orchestration boundaries: synthetic transports, no SDK, network or NPU."""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import host, profiling, profiling_export
from tests import test_host as fixtures


def profile_args(build, temporary):
    parser = temporary / "fake-msprof.py"
    parser.write_text("# Synthetic exporter placeholder; tests mock its invocation.\n")
    return SimpleNamespace(
        build=build,
        host="fake-1952",
        remote_dir="/tmp/pypto-profile-test/session",
        runtime_env=None,
        python=sys.executable,
        device=0,
        timeout=30,
        msprof="/var/msprof",
        msprof_py=parser,
        warmup=1,
        repeat=2,
    )


def add_profile_result(fixture, *, bad_final=False):
    final = fixture.results / "profile_actual.bin"
    raw = (fixture.case_dir / "z.bin").read_bytes()
    final.write_bytes(bytes(len(raw)) if bad_final else raw)
    report = fixture.report()
    report["profiling_execution"] = {
        "enabled": True,
        "status": "completed",
        "warmup": 1,
        "repeat": 2,
        "completed_warmup": 1,
        "completed_repeat": 2,
        "correctness_calls": 2,
        "total_execution_calls": 5,
        "attempted_execution_calls": 5,
        "discard_first_executions": 3,
        "host_wall_seconds": [0.00002, 0.00003],
        "timing_scope": "synchronous_aclmdlExecute_host_wall",
        "final_output": {"file": final.name, "sha256": host.sha256(final), "bytes": len(raw)},
    }
    fixture.write_report(report)


def check_profile_transfer_and_acceptance_boundaries(tmp_path, mode):
    fixture = fixtures.HostTests()
    fixture.setUp()
    try:
        audit = fixture.build / "atc/om_audit.json"
        audit.parent.mkdir()
        audit.write_text(
            json.dumps({"om_sha256": fixture.binding["om_sha256"], "graph": {"kernels": [{"name": "native_pro"}]}})
        )
        fixture.binding["om_audit_sha256"] = host.sha256(audit)
        fixture.binding_path.write_text(json.dumps(fixture.binding))
        # A valid older execution-only bundle can retain its OM and inputs.
        old_runner = fixture.bundle / "runner.py"
        old_runner.write_text("# Previous execution-only runner.\n")
        inventory_path = fixture.bundle / "bundle.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["files"]["runner.py"] = host.sha256(old_runner)
        inventory_path.write_text(json.dumps(inventory))
        add_profile_result(fixture, bad_final=mode == "bad_final")
        args = profile_args(fixture.build, tmp_path)
        output = tmp_path / "output"
        output.mkdir()
        calls = []

        def transport(command, **kwargs):
            calls.append(command)
            assert kwargs["timeout"] == args.timeout and kwargs["check"] is False
            rc = 0
            if command[0] == "scp" and command[2].startswith("fake-1952:"):
                destination = Path(command[3])
                if command[2].endswith("/results"):
                    if mode == "download_failure":
                        rc = 47
                    else:
                        shutil.copytree(fixture.results, destination)
                else:
                    destination.mkdir()
                    state = json.loads((output / "profile_state.json").read_text())
                    names = [] if mode == "no_prof" else ["PROF_synthetic"]
                    if mode == "multi_prof":
                        names.append("PROF_second")
                    for name in names:
                        directory = destination / name
                        directory.mkdir()
                        (directory / "raw.synthetic").write_bytes(b"not real profiling data")
                    capture = {
                        "schema_version": 1,
                        "run_id": "stale" if mode == "wrong_run_id" else state["run_id"],
                        "runner_sha256": state["runner_sha256"],
                        "environment_rc": 0,
                        "runner_rc": 37 if mode == "swallowed_child_failure" else 0,
                        "profiler_rc": 19 if mode == "profiler_failure" else 0,
                        "collection_rc": 0 if len(names) == 1 else 2,
                        "prof": names[0] if len(names) == 1 else None,
                        "prof_directories": names,
                    }
                    (destination / "capture.json").write_text(json.dumps(capture))
            # Even an outer zero cannot override independent child-RC evidence.
            return SimpleNamespace(returncode=rc, stdout="synthetic transport", stderr="")

        with (
            patch.object(profiling.subprocess, "run", side_effect=transport),
            patch.object(
                profiling_export, "export_and_parse", return_value={"source": "synthetic test only"}
            ) as exporter,
        ):
            if mode == "export_failure":
                exporter.side_effect = RuntimeError("synthetic export failure")
            report = profiling.remote_profile(args, output)
        assert [command[0] for command in calls] == ["ssh", "scp", "ssh", "scp", "scp"]
        assert calls[3][2].endswith("/payload/results")
        assert calls[4][2].endswith("/payload/capture")
        assert (output / "raw/capture.json").is_file()
        assert report["status"] == ("passed" if mode == "success" else "failed")
        assert (report["performance"] is not None) == (mode == "success")
        assert exporter.call_count == (1 if mode in ("success", "export_failure") else 0)
        if mode != "download_failure":
            assert (output / "board/actual_second.bin").is_file()
        staged = output / "staging/bundle"
        assert host.sha256(staged / "runner.py") == host.sha256(profiling.HERE / "runner.py")
        assert host.sha256(staged / "runner.py") != host.sha256(old_runner)
        assert not any((staged / name).exists() for name in ("case.json", "binding.json", "z.bin"))
        state = json.loads((output / "profile_state.json").read_text())
        assert state["status"] == report["status"]
        if mode == "swallowed_child_failure":
            assert state["capture"]["runner_rc"] == 37 and state["capture"]["profiler_rc"] == 0
            assert "runner_rc=37" in report["errors"]
        if mode == "bad_final":
            assert all(item["passed"] for item in report["accuracy"]["comparisons"])
            assert report["accuracy"]["profile_final_comparison"]["passed"] is False
    finally:
        fixture.doCleanups()


def check_real_generated_shell_records_child_rc_even_when_profiler_swallows_it(tmp_path):
    payload = tmp_path / "payload"
    (payload / "bundle").mkdir(parents=True)
    (payload / "bundle/runner.py").write_text("raise SystemExit(37)\n")
    profiler = tmp_path / "fake-msprof"
    profiler.write_text('#!/bin/bash\nset +e\n"$@"\nmkdir PROF_synthetic\nexit 0\n')
    profiler.chmod(0o755)
    environment = tmp_path / "runtime.sh"
    environment.write_text("set -e\n")
    args = SimpleNamespace(
        python=sys.executable, device=0, warmup=1, repeat=2, runtime_env=str(environment), msprof=str(profiler)
    )
    application, collector = profiling.remote_scripts(args, "synthetic-run")
    (payload / "app.sh").write_text(application)
    (payload / "profile.sh").write_text(collector)
    process = subprocess.run(
        ["bash", str(payload / "profile.sh")], capture_output=True, text=True, timeout=20, check=False
    )
    capture = json.loads((payload / "capture/capture.json").read_text())
    assert process.returncode == 1
    assert capture["runner_rc"] == 37 and capture["profiler_rc"] == 0
    assert capture["environment_rc"] == capture["collection_rc"] == 0
    assert capture["run_id"] == "synthetic-run" and capture["prof"] == "PROF_synthetic"
    assert (payload / "runner.rc").read_text().strip() == "37"


class ProfilingTests(unittest.TestCase):
    def test_fp32_final_precision_is_compared_on_host_using_streaming_io(self):
        fixture = fixtures.HostTests()
        fixture.setUp()
        try:
            fixture.use_fp32()
            add_profile_result(fixture)
            args = profile_args(fixture.build, fixture.root)
            with (
                patch.object(Path, "read_bytes", side_effect=AssertionError("No whole-tensor reads")),
                patch.object(host, "IO_CHUNK_BYTES", 5),
            ):
                execution, comparison = profiling.validate_profile_execution(fixture.build, fixture.results, args)
            self.assertEqual(execution["completed_repeat"], 2)
            self.assertTrue(comparison["passed"])
            self.assertEqual(comparison["elements"], 4)
            final = fixture.results / "profile_actual.bin"
            final.write_bytes(struct.pack("<ffff", 6, 8, 10, float("nan")))
            report = fixture.report()
            report["profiling_execution"]["final_output"]["sha256"] = host.sha256(final)
            fixture.write_report(report)
            _, comparison = profiling.validate_profile_execution(fixture.build, fixture.results, args)
            self.assertFalse(comparison["passed"])
            self.assertEqual(comparison["nonfinite"], 1)
        finally:
            fixture.doCleanups()

    def test_transfer_and_acceptance_boundaries(self):
        modes = (
            "success",
            "swallowed_child_failure",
            "profiler_failure",
            "no_prof",
            "multi_prof",
            "wrong_run_id",
            "download_failure",
            "bad_final",
            "export_failure",
        )
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                check_profile_transfer_and_acceptance_boundaries(Path(temporary), mode)

    @unittest.skipUnless(shutil.which("bash"), "generated board scripts require bash")
    def test_real_generated_shell_records_swallowed_child_return_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            check_real_generated_shell_records_child_rc_even_when_profiler_swallows_it(Path(temporary))


if __name__ == "__main__":
    unittest.main()
