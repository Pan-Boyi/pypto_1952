"""One-command orchestration and presentation; board/timing data are synthetic."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pypto_1952 import dec_enc_profile as profile


def statistics(value):
    return {key: value for key in ("min", "median", "mean", "max")}


def result():
    return {
        "status": "passed",
        "om_sha256": "1" * 64,
        "raw_profiling": "/recovered/PROF_synthetic",
        "accuracy": {"status": "passed", "profile_final_comparison": {"passed": True}},
        "performance": {
            "device_units": "us",
            "repeat": 2,
            "expected_nodes": ["audited_dec_enc"],
            "statistics_csv": {"path": "/analysis/node_statistics.csv"},
            "nodes": {
                "audited_dec_enc": {
                    "statistics_us": statistics(42),
                    "pipeline_us": {
                        key: {"status": "available", "statistics": statistics(index)}
                        for index, (_, key) in enumerate(profile.PIPELINES)
                    },
                }
            },
        },
    }


class OneClickTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.config_path = self.root / "settings.json"
        self.toolkit = self.root / "sdk"
        self.toolkit.mkdir()
        (self.toolkit / "set_env.sh").write_text("export PYPTO_FAKE_SELECTED_SDK=yes\n")
        self.pto = self.root / "pto-isa"
        (self.pto / "include/pto").mkdir(parents=True)
        (self.pto / "include/pto/pto-inst.hpp").write_text("// Synthetic headers\n")
        self.build = self.repo / "build"
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.repo) + "\n")
        self.parser = self.toolkit / "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        self.parser.parent.mkdir(parents=True)
        self.parser.write_text("# Synthetic parser\n")
        self.raw = {
            "toolkit_root": "sdk",
            "pypto_root": "checkout",
            "gym_root": str(Path(__file__).resolve().parents[2] / "pypto-gym"),
            "pto_root": "pto-isa",
            "host": "test-board",
            "output_root": "runs",
            "runtime_env": "/board/runtime/set_env.sh",
            "repeat": 2,
        }
        self.config_path.write_text(json.dumps(self.raw))

    def config(self, **changes):
        self.config_path.write_text(json.dumps({**self.raw, **changes}))
        with patch.object(profile, "REPO", self.repo):
            return profile.load_config(self.config_path)

    def test_shape_short_and_full_and_rejected_geometry(self):
        for spelling in ("2160x3840", "1x4x2160x3840", "2160,3840", "2160×3840"):
            self.assertEqual(profile.parse_shape(spelling), [2160, 3840])
        for spelling in ("2x4x2160x3840", "1x3x540x960", "540x", "$(touch bad)", "540 960"):
            with self.subTest(spelling=spelling), self.assertRaises(ValueError):
                profile.parse_shape(spelling)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(profile.main(["fp16", "541x960", "--config", str(self.config_path)]), 1)

    def test_config_resolves_relative_paths_and_deterministic_auto_paths(self):
        config = self.config()
        self.assertEqual(config["toolkit_root"], self.toolkit)
        self.assertEqual(config["build_dir"], self.build)
        self.assertEqual(config["msprof_py"], self.parser)
        self.assertEqual(config["output_root"], self.root / "runs")
        self.assertEqual(config["pto_root"], self.pto)

    def test_environment_sdk_selection_and_explicit_sdk_priority(self):
        self.config_path.write_text(json.dumps({**self.raw, "toolkit_root": None}))
        with (
            patch.object(profile, "REPO", self.repo),
            patch.dict(os.environ, {"PYPTO_PRO_TOOLKIT_ROOT": str(self.toolkit)}),
        ):
            self.assertEqual(profile.load_config(self.config_path)["toolkit_root"], self.toolkit)
        with patch.dict(os.environ, {"PYPTO_PRO_TOOLKIT_ROOT": "/bad-sdk"}):
            self.assertEqual(self.config()["toolkit_root"], self.toolkit)

    def test_rejects_wrong_build_source_unknown_fields_and_remote_roots(self):
        for changes in (
            {"password": "must-not-be-persisted"},
            {"repeat": 0},
            {"warmup": True},
            {"timeout": -1},
            {"runtime_env": "relative.sh"},
            {"msprof": "relative-msprof"},
            {"host": "-bad-host"},
            {"remote_root": "/"},
            {"remote_root": "/tmp/task;rm"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        (self.build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/different-checkout\n")
        with self.assertRaisesRegex(ValueError, "build_dir"):
            self.config()

    def test_init_is_explicit_and_never_overwrites(self):
        path = self.root / "new/settings.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(profile.main(["--init", "--config", str(path)]), 0)
            original = path.read_bytes()
            self.assertEqual(profile.main(["--init", "--config", str(path)]), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(json.loads(path.read_text()), profile.DEFAULTS)

    def test_sdk_is_sourced_in_same_child_preserving_selected_python_and_checkout(self):
        config = self.config()
        with patch.object(profile, "REPO", self.repo):
            command = profile._compiler_command(config, ["build"])
        prefix = command.index(sys.executable)
        code = (
            "import os; print(os.environ['PYPTO_FAKE_SELECTED_SDK']); print(os.environ['PYTHONPATH']); "
            "print(os.environ['PATH'].split(os.pathsep)[0])"
        )
        executed = subprocess.run([*command[: prefix + 1], "-c", code], text=True, capture_output=True, check=True)
        self.assertEqual(executed.stdout.splitlines()[0], "yes")
        self.assertTrue(
            executed.stdout.splitlines()[1].startswith(str(self.repo / "python") + os.pathsep + str(self.repo))
        )
        self.assertEqual(executed.stdout.splitlines()[2], str(Path(sys.executable).parent))

    def test_all_pipeline_rows_and_actual_zero_and_missing_values(self):
        report = result()
        report["performance"]["nodes"]["audited_dec_enc"]["pipeline_us"]["fixpipe"] = {
            "status": "not_available",
            "statistics": statistics(999),
        }
        text = profile.format_summary(report, "fp16", [2160, 3840])
        for name in ("Task Duration", "cube", "vec", "mte1", "mte2", "mte3", "fixpipe", "scalar"):
            self.assertIn(name, text)
        self.assertIn("cube                    0.000", text)
        self.assertIn("fixpipe        未提供 (N/A)", text)
        self.assertNotIn("999.000", text)
        self.assertIn("不能相加", text)

    def test_bad_accuracy_units_and_nonfinite_statistics_withhold_summary(self):
        reports = []
        for status in ("failed", "started"):
            reports.append({**result(), "status": status})
        wrong = result()
        wrong["accuracy"]["profile_final_comparison"]["passed"] = False
        reports.append(wrong)
        wrong = result()
        wrong["performance"]["device_units"] = "cycles"
        reports.append(wrong)
        wrong = result()
        wrong["performance"]["nodes"]["audited_dec_enc"]["statistics_us"] = statistics(float("nan"))
        reports.append(wrong)
        for report in reports:
            with self.subTest(report=report), self.assertRaises(ValueError):
                profile.format_summary(report, "fp16", [540, 960])

    def run_fake(self, config, *, failure=None, report=None):
        commands = []

        def run(command, log, **kwargs):
            commands.append(command)
            log.write_text("Synthetic invocation\n")
            if len(commands) == 1:
                self.assertEqual(command[0], "ssh")
                self.assertIn("BatchMode=yes", command)
                if failure == "preflight":
                    raise RuntimeError("no board connection")
                return
            phase = "build" if "build" in command else "profile"
            if failure == phase:
                raise RuntimeError(phase + " failed")
            output = Path(command[command.index("--output") + 1])
            self.assertFalse(output.exists())
            output.mkdir()
            if phase == "build":
                (output / "build_result.json").write_text(json.dumps({"status": "bundle-built"}))
            else:
                (output / "result.json").write_text(json.dumps(report or result()))

        with patch.object(profile, "_run", side_effect=run), redirect_stdout(io.StringIO()) as stdout:
            with redirect_stderr(io.StringIO()):
                rc = profile.execute(config, "fp16", 2160, 3840)
        return rc, commands, stdout.getvalue()

    def test_one_command_build_then_profile_then_all_pipeline_summary(self):
        config = self.config()
        rc, commands, output = self.run_fake(config)
        self.assertEqual(rc, 0)
        self.assertEqual(len(commands), 3)
        build, profiler = commands[1:]
        self.assertIn("--build-dir", build)
        self.assertIn("pipelined", build)
        self.assertIn("fp16", build)
        self.assertNotIn("--rows-per-tile", build)  # Real build resolves fp16=12/fp32=6.
        self.assertIn("--runtime-env", profiler)
        root = next(config["output_root"].iterdir())
        state = json.loads((root / "one_click_result.json").read_text())
        self.assertEqual(state["status"], "passed")
        self.assertEqual(state["configuration"]["rows"], 12)
        self.assertEqual(state["configuration"]["block_dim"], 8)
        self.assertTrue((root / "summary.txt").exists())
        self.assertIn("fixpipe", output)
        self.assertIn("scalar", output)

    def test_failed_stage_stops_later_stages_and_retains_failure(self):
        for phase, count in (("preflight", 1), ("build", 2), ("profile", 3)):
            config = self.config(output_root=str(self.root / phase))
            rc, commands, output = self.run_fake(config, failure=phase)
            self.assertEqual(rc, 1)
            self.assertEqual(len(commands), count)
            root = next(config["output_root"].iterdir())
            state = json.loads((root / "one_click_result.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], phase)
            self.assertFalse((root / "summary.txt").exists())
            self.assertNotIn("精度: PASSED", output)

    def test_failed_precision_and_same_case_rerun_use_distinct_directories(self):
        config = self.config()
        failed = copy.deepcopy(result())
        failed["status"] = "failed"
        self.assertEqual(self.run_fake(config, report=failed)[0], 1)
        self.assertEqual(self.run_fake(config)[0], 0)
        self.assertEqual(len(list(config["output_root"].iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
