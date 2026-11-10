"""Matrix orchestration retains per-case failure and requires explicit host paths."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import dec_enc_matrix as matrix


class MatrixTests(unittest.TestCase):
    def test_full_matrix_contains_all_shape_dtype_pairs(self):
        configs = matrix.cases("full", 8, 3)
        self.assertEqual(len(configs), 32)
        self.assertEqual(len({(c["height"], c["width"], c["dtype"]) for c in configs}), 32)

    def test_failure_is_recorded_and_remaining_cases_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(
                matrix.subprocess, "run", side_effect=[SimpleNamespace(returncode=i) for i in (0, 1, 0, 0)]
            ) as run:
                status = matrix.main(
                    ["--phase", "build", "--root", directory, "--toolkit-root", "/sdk", "--pto-root", "/pto"]
                )
            self.assertEqual(status, 1)
            self.assertEqual(run.call_count, 4)
            report = json.loads((Path(directory) / "matrix_build.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual([c["returncode"] for c in report["cases"]], [0, 1, 0, 0])

    def test_profile_uses_compiler_parser_and_fresh_per_case_remote_paths(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(matrix.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(
                    matrix.main(
                        [
                            "--phase",
                            "profile",
                            "--root",
                            directory,
                            "--host",
                            "board5102",
                            "--remote-root",
                            "/tmp/dec-enc",
                            "--msprof-py",
                            "/compiler/analysis/msprof/msprof.py",
                        ]
                    ),
                    0,
                )
            commands = [call.args[0] for call in run.call_args_list]
            remote = [cmd[cmd.index("--remote-dir") + 1] for cmd in commands]
            self.assertEqual(len(set(remote)), 4)
            self.assertTrue(all("--msprof-py" in cmd and "/var/msprof" in cmd for cmd in commands))

    def test_missing_parser_and_results_path_traversal_are_rejected(self):
        for extra in ([], ["--msprof-py", "/parser.py", "--results-tag", "../bad"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                matrix.main(
                    ["--phase", "profile", "--root", "/unused", "--host", "board", "--remote-root", "/tmp/task", *extra]
                )
            self.assertEqual(error.exception.code, 2)

    def test_baseline_build_selection_is_explicit_and_has_distinct_case_paths(self):
        commands = []
        for schedule in ("baseline", "optimized", "pipelined"):
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                with patch.object(matrix.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                    self.assertEqual(
                        matrix.main(
                            [
                                "--phase",
                                "build",
                                "--root",
                                directory,
                                "--toolkit-root",
                                "/sdk",
                                "--pto-root",
                                "/pto",
                                "--dec-enc-schedule",
                                schedule,
                            ]
                        ),
                        0,
                    )
                    command = run.call_args_list[0].args[0]
                    self.assertEqual(command[command.index("--dec-enc-schedule") + 1], schedule)
                    self.assertIn("_" + schedule, command[command.index("--output") + 1])
                    commands.append(command)
        self.assertNotEqual(commands[0], commands[1])

    def test_pipeline_defaults_match_reference_dtype_buffer_budget(self):
        configs = matrix.cases("smoke", 8, None, "pipelined")
        self.assertEqual([config["rows"] for config in configs], [6, 12, 6, 12])
        self.assertTrue(all(config["rows"] == 3 for config in matrix.cases("smoke", 8, None, "optimized")))


if __name__ == "__main__":
    unittest.main()
