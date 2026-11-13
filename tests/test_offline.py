"""Offline archive boundaries with real packaging/runner/evaluation and synthetic ACL only."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import ctypes as ct
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tarfile
import unittest
from unittest.mock import patch

from pypto_1952 import host, offline, offline_run, runner
from tests import test_host as fixtures
from tests.test_runner import FakeACL


class ArchiveACL(FakeACL):
    """Synthetic 2x2 Add for the host acceptance fixture, including a non-writing mode."""

    def aclmdlGetInputDims(self, _, index, dimensions):
        dimensions._obj.dimCount = 2
        dimensions._obj.dims[0] = dimensions._obj.dims[1] = 2
        return 0

    aclmdlGetOutputDims = aclmdlGetInputDims

    def aclmdlGetInputSizeByIndex(self, *_):
        return 8

    aclmdlGetOutputSizeByIndex = aclmdlGetInputSizeByIndex

    def aclmdlExecute(self, _, inputs, outputs):
        self.executions += 1
        if self.mode == "no_write":
            return 0
        values = [struct.unpack("<4e", ct.string_at(*self.buffers[b])) for b in self.datasets[inputs]]
        payload = struct.pack("<4e", *[a + b for a, b in zip(*values)])
        address, _ = self.buffers[self.datasets[outputs][0]]
        ct.memmove(address, payload, len(payload))
        return 0


class OfflineTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.HostTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root.resolve()
        (fixture.build / "build_result.json").write_text(json.dumps({
            "status": "bundle-built", "target_soc": "MC62CM12AA", "device_execution": "not_run"}))
        self.package = self.root / "package"
        with redirect_stdout(io.StringIO()):
            offline.package(fixture.build, self.package)
        # Transfer only runtime.tar.gz to a directory without the repos or golden data.
        self.board = self.root / "board"
        self.board.mkdir()
        with tarfile.open(self.package / "runtime.tar.gz") as archive:
            # This archive was produced above from the trusted test fixture.
            archive.extractall(self.board)
        self.runtime = self.board / "runtime"

    def board_execute(self, fake_acl=None, repeat=0, warmup=0):
        self.result = self.board / "job"
        command = ["--output", str(self.result), "--repeat", str(repeat), "--warmup", str(warmup)]
        real_popen = subprocess.Popen

        def launch(argv, **options):
            # The actual runner executes with a FakeACL; keep the board wrapper's child-process boundary.
            arguments = argv[argv.index("--execution"):]
            with patch.object(runner, "load_acl", return_value=fake_acl or ArchiveACL()), \
                    redirect_stdout(io.StringIO()) as log:
                rc = runner.main(arguments)
            return type("Child", (), {"stdout": io.StringIO(log.getvalue()), "wait": lambda self: rc})()

        with patch.object(offline_run, "__file__", str(self.runtime / "offline_run.py")), \
                patch.object(offline_run.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            rc = offline_run.main(command)
        self.assertIs(subprocess.Popen, real_popen)
        return rc

    def accept(self):
        self.accepted = self.root / "accepted"
        with redirect_stdout(io.StringIO()):
            rc = offline.accept(self.fixture.build, self.result / "results.tar.gz", self.accepted)
        return rc, json.loads((self.accepted / "result.json").read_text())

    def test_portable_package_contains_no_golden_or_build_dependencies(self):
        manifest = offline_run.verify(self.runtime)
        self.assertEqual(manifest["device_execution"], "not_run")
        self.assertEqual(manifest["om_sha256"], self.fixture.binding["om_sha256"])
        self.assertEqual(host.sha256(self.package / "runtime.tar.gz"),
                         json.loads((self.package / "package_result.json").read_text())["archive"]["sha256"])
        self.assertFalse((self.runtime / "bundle/z.bin").exists())
        for path in self.runtime.rglob("*.py"):
            ast.parse(path.read_text(), feature_version=(3, 7))
        child = subprocess.run([sys.executable, "-I", "-S", "-B", str(self.runtime / "offline_run.py"),
                                "--verify-only"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["status"], "verified")

    def test_return_archive_is_accepted_only_after_compiler_comparison(self):
        self.assertEqual(self.board_execute(), 0)
        board_report = json.loads((self.result / "board_run.json").read_text())
        self.assertEqual(board_report["status"], "executed")
        self.assertNotIn("passed", board_report.values())
        rc, report = self.accept()
        self.assertEqual(rc, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["precision_checked_on"], "compiler_host")
        offline_run.verify(self.runtime)  # Execution never mutates the runtime tree.

    def test_optional_host_wall_samples_and_final_output_are_checked(self):
        self.assertEqual(self.board_execute(repeat=3, warmup=1), 0)
        rc, report = self.accept()
        self.assertEqual(rc, 0)
        self.assertTrue(report["profile_final_comparison"]["passed"])
        self.assertEqual(report["whole_om_host_wall_us"]["count"], 3)
        self.assertIn("not_device_kernel_time", report["whole_om_host_wall_us"]["scope"])

    def test_nonwriting_board_kernel_is_rejected_at_acceptance(self):
        self.assertEqual(self.board_execute(ArchiveACL(mode="no_write")), 0)
        rc, report = self.accept()
        self.assertEqual(rc, 1)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["dual_sentinel_consistent"])

    def test_board_failure_still_returns_diagnostics_and_nonzero(self):
        self.assertEqual(self.board_execute(FakeACL(mismatch="count")), 1)
        self.assertTrue((self.result / "results.tar.gz").is_file())
        rc, report = self.accept()
        self.assertEqual(rc, 1)
        self.assertIn("Board failed", report["error"])
        self.assertTrue((self.accepted / "returned/runner.log").is_file())

    def test_corrupted_runtime_fails_before_npu_load(self):
        (self.runtime / "bundle/model.om").write_bytes(b"changed")
        with patch.object(offline_run, "__file__", str(self.runtime / "offline_run.py")), \
                patch.object(offline_run.subprocess, "Popen") as launch, redirect_stderr(io.StringIO()):
            self.assertEqual(offline_run.main(["--output", str(self.board / "job")]), 1)
            launch.assert_not_called()

    def test_result_inside_runtime_or_reused_output_is_rejected(self):
        with patch.object(offline_run, "__file__", str(self.runtime / "offline_run.py")), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(offline_run.main(["--output", str(self.runtime / "results")]), 1)
            self.assertEqual(offline_run.main(["--output", str(self.board)]), 1)
        with self.assertRaises(FileExistsError):
            offline.package(self.fixture.build, self.package)

    def test_archive_traversal_is_rejected(self):
        archive = self.root / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("../escaped")
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(offline.accept(self.fixture.build, archive, self.root / "bad-accept"), 1)
        self.assertFalse((self.root / "escaped").exists())

    def test_build_uses_existing_provider_compiler_before_packaging(self):
        from pypto_1952 import run_profile

        args = ["build", "--op", "example", "--output", str(self.root / "new-build")]
        with patch.object(run_profile, "main", return_value=0) as build, patch.object(offline, "package") as pack:
            self.assertEqual(offline.main(args), 0)
            build.assert_called_once_with(args)
            pack.assert_called_once_with(self.root / "new-build", self.root / "new-build/offline")
        with patch.object(run_profile, "main", return_value=17), patch.object(offline, "package") as pack:
            self.assertEqual(offline.main(args), 17)
            pack.assert_not_called()

    def test_shell_preserves_selected_python_and_resolves_operator_from_any_cwd(self):
        repo = Path(__file__).resolve().parents[1]
        selected = self.root / "selected python"
        args_file = self.root / "args.txt"
        selected.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$OFFLINE_TEST_ARGS"\n')
        selected.chmod(0o755)
        environment = dict(os.environ, PYTHON=str(selected), OFFLINE_TEST_ARGS=str(args_file),
                           PYPTO_1952_ROOT=str(repo), PYPTO_PRO_TOOLKIT_ROOT="/sdk path", PTO_LIB_PATH="/pto path")
        gym = repo.parent / "pypto-gym/ops/pro/experimental/gemm_fused_kernel"
        command = ["bash", str(gym / "scripts/test_static_1952.sh"), "build", "--output", "/new build"]
        result = subprocess.run(command, cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = args_file.read_text().splitlines()
        self.assertEqual(arguments[:5], [str(repo), "-B", "-m", "pypto_1952.offline", "build"])
        self.assertEqual(arguments[arguments.index("--provider") + 1], str(gym / "provider.py"))
        self.assertEqual(arguments[arguments.index("--toolkit-root") + 1], "/sdk path")
        self.assertEqual(arguments[arguments.index("--output") + 1], "/new build")

    def test_failed_board_environment_cannot_be_swallowed_by_set_plus_e(self):
        environment = self.root / "bad runtime.sh"
        environment.write_text("set +e\nfalse\n")
        result = subprocess.run(["bash", str(self.runtime / "run.sh"), "--runtime-env", str(environment),
                                 "--verify-only"], cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Board runtime environment failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
