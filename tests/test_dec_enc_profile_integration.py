"""Synthetic one-command integration; no compiler, SSH connection or NPU measurements."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import dec_enc_profile, host, profiling, run_profile
from tests import test_host as fixtures
from tests.test_profiling import add_profile_result

NODE = "audited_synthetic_dec_enc"
PIPELINES = {
    "cube": "Cube_Exe_Time (us)",
    "vector": "Vec_Time (us)",
    "mte1": "MTE1 Exe Time (us)",
    "mte2": "aic_mte2_time(us)",
    "mte3": "mte3_time(us)",
    "fixpipe": "fixpipe_exe_time(us)",
    "scalar": "aic_scalar_time(us)",
}


class DecEncProfileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HostTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.resolve()
        self.fixture.case["operator"] = "dec_enc_fused_kernel"
        self.fixture.case_path.write_text(json.dumps(self.fixture.case))
        self.fixture.binding["case_sha256"] = host.sha256(self.fixture.case_path)
        audit = self.fixture.build / "atc/om_audit.json"
        audit.parent.mkdir()
        audit.write_text(
            json.dumps(
                {
                    "om_sha256": self.fixture.binding["om_sha256"],
                    "graph": {"kernels": [{"name": NODE}]},
                }
            )
        )
        self.fixture.binding["om_audit_sha256"] = host.sha256(audit)
        self.fixture.binding_path.write_text(json.dumps(self.fixture.binding))
        self.parser = self.root / "fake-suite/analysis/msprof/msprof.py"
        self.parser.parent.mkdir(parents=True)
        self.config = {
            **dec_enc_profile.DEFAULTS,
            "toolkit_root": self.root / "fake-sdk",
            "pypto_root": self.root / "fake-pypto",
            "gym_root": Path(__file__).resolve().parents[2] / "pypto-gym",
            "pto_root": self.root / "fake-pto",
            "build_dir": self.root / "fake-cmake",
            "output_root": self.root / "runs",
            "host": "synthetic-1952",
            "remote_root": "/tmp/synthetic-dec-enc-integration",
            "python": sys.executable,
            "msprof_py": self.parser,
            "warmup": 1,
            "repeat": 2,
            "timeout": 20,
        }

    def exporter(self, *, fail=False):
        rows = []
        for execution in range(5):
            row = {
                "Op Name": NODE,
                "Task Start Time(us)": 100 * execution,
                "Task Duration(us)": 900001 + execution if execution < 3 else 40 + (execution - 3) * 4,
            }
            for index, column in enumerate(PIPELINES.values()):
                row[column] = 900001 + execution if execution < 3 else index * 2 + (execution - 3) * 2
            # A real zero must remain distinguishable from an absent pipeline metric.
            if execution >= 3:
                row[PIPELINES["cube"]] = 0
            rows.append(row)
        (self.parser.parents[2] / "fixture_rows.json").write_text(json.dumps(rows))
        self.parser.write_text(
            "import csv,json,sys\nfrom pathlib import Path\n"
            "print('SYNTHETIC EXPORTER: no hardware performance measurements')\n"
            "assert sys.argv[1:4] == ['export','summary','-dir']\n"
            "assert Path.cwd() == Path(__file__).resolve().parents[2]\n"
            f"if {fail!r}:\n print('synthetic export failure',file=sys.stderr)\n sys.exit(17)\n"
            "rows=json.loads(Path('fixture_rows.json').read_text())\n"
            "output=Path(sys.argv[4])/'mindstudio_profiler_output'\noutput.mkdir()\n"
            "with (output/'op_summary_synthetic.csv').open('w',newline='') as stream:\n"
            " writer=csv.DictWriter(stream,fieldnames=list(rows[0]))\n"
            " writer.writeheader()\n writer.writerows(rows)\n"
        )

    def execute(self, *, bad_final=False, export_failure=False):
        self.exporter(fail=export_failure)
        add_profile_result(self.fixture, bad_final=bad_final)
        self.transport_calls, self.export_calls, self.stage_calls = [], [], []
        real_subprocess_run = subprocess.run

        def transport(command, **kwargs):
            if command[0] not in ("ssh", "scp"):
                self.assertEqual(command[:4], [sys.executable, str(self.parser), "export", "summary"])
                self.export_calls.append(command)
                # The actual exporter child writes its CSV and logs; do not invent a performance JSON.
                return real_subprocess_run(command, **kwargs)
            self.transport_calls.append(command)
            self.assertEqual(kwargs["timeout"], 20)
            self.assertFalse(kwargs["check"])
            if command[0] == "scp" and command[2].startswith(self.config["host"] + ":"):
                destination = Path(command[3])
                if command[2].endswith("/results"):
                    shutil.copytree(self.fixture.results, destination)
                else:
                    destination.mkdir()
                    prof = destination / "PROF_synthetic"
                    prof.mkdir()
                    (prof / "raw.synthetic").write_bytes(b"synthetic transport evidence; not profiler data")
                    state = json.loads((destination.parent / "profile_state.json").read_text())
                    capture = {
                        "schema_version": 1,
                        "run_id": state["run_id"],
                        "runner_sha256": state["runner_sha256"],
                        "environment_rc": 0,
                        "runner_rc": 0,
                        "profiler_rc": 0,
                        "collection_rc": 0,
                        "prof": prof.name,
                        "prof_directories": [prof.name],
                    }
                    (destination / "capture.json").write_text(json.dumps(capture))
            return SimpleNamespace(returncode=0, stdout="synthetic transport only", stderr="")

        def stage(command, log, **_kwargs):
            self.stage_calls.append(command)
            if command[0] == "ssh":
                log.write_text("Synthetic preflight; no board connected\n")
                return
            arguments = command[command.index("-m") + 2 :]
            output = Path(arguments[arguments.index("--output") + 1])
            self.assertFalse(output.exists())
            if arguments[0] == "build":
                shutil.copytree(self.fixture.build, output)
                (output / "build_result.json").write_text(json.dumps({"status": "bundle-built"}))
                log.write_text("Synthetic build copied a small structural fixture; no native compilation\n")
                return
            self.assertEqual(arguments[0], "profile")
            with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
                rc = run_profile.main(arguments)
            if rc:
                raise RuntimeError("阶段返回码 {}；日志: {}".format(rc, log))

        with (
            patch.object(dec_enc_profile, "_run", side_effect=stage),
            patch.object(profiling.subprocess, "run", side_effect=transport),
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            rc = dec_enc_profile.execute(self.config, "fp16", 540, 960)
        root = next(self.config["output_root"].iterdir())
        self.assertEqual(len(self.stage_calls), 3)
        self.assertEqual([command[0] for command in self.transport_calls], ["ssh", "scp", "ssh", "scp", "scp"])
        self.assertTrue((root / "profile/raw/PROF_synthetic/raw.synthetic").is_file())
        self.assertTrue((root / "profile/board/actual_second.bin").is_file())
        return rc, root, stdout.getvalue(), stderr.getvalue()

    def test_real_export_parse_and_print_all_seven_measured_pipelines(self):
        rc, root, stdout, _ = self.execute()
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.export_calls), 1)
        report = json.loads((root / "profile/result.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["accuracy"]["profile_final_comparison"]["passed"])
        performance = report["performance"]
        self.assertEqual(performance["device_units"], "us")
        self.assertEqual(performance["execution_count_per_node"], 5)
        self.assertEqual(performance["repeat"], 2)
        node = performance["nodes"][NODE]
        self.assertEqual(node["correctness_prefix_us"], [900001, 900002])
        self.assertEqual(node["warmup_us"], [900003])
        self.assertEqual(node["measured_us"], [40, 44])
        self.assertEqual(node["statistics_us"], {"min": 40, "median": 42, "mean": 42, "max": 44})
        for index, metric in enumerate(PIPELINES):
            value = node["pipeline_us"][metric]
            expected = [0, 0] if metric == "cube" else [index * 2, index * 2 + 2]
            self.assertEqual(value["status"], "available")
            self.assertEqual(value["samples"], expected)
            self.assertEqual(value["statistics"]["median"], sum(expected) / 2)
        for label in ("Task Duration", "cube", "vec", "mte1", "mte2", "mte3", "fixpipe", "scalar"):
            self.assertIn(label, stdout)
        self.assertIn("Task Duration          42.000", stdout)
        self.assertIn("cube                    0.000", stdout)
        self.assertNotIn("90000", (root / "summary.txt").read_text())
        self.assertEqual(json.loads((root / "one_click_result.json").read_text())["status"], "passed")
        with Path(performance["statistics_csv"]["path"]).open(newline="") as stream:
            statistics = list(csv.DictReader(stream))
        task = next(row for row in statistics if row["metric"] == "task_duration")
        self.assertEqual((task["node"], task["units"], task["samples"], float(task["median"])), (NODE, "us", "2", 42))
        export_log = root / "profile/analysis/export_stdout.log"
        self.assertIn("SYNTHETIC EXPORTER", export_log.read_text())

    def test_final_precision_failure_withholds_export_and_summary(self):
        rc, root, stdout, stderr = self.execute(bad_final=True)
        self.assertEqual(rc, 1)
        self.assertEqual(self.export_calls, [])
        report = json.loads((root / "profile/result.json").read_text())
        self.assertTrue(all(item["passed"] for item in report["accuracy"]["comparisons"]))
        self.assertFalse(report["accuracy"]["profile_final_comparison"]["passed"])
        self.assertIsNone(report["performance"])
        self.assertFalse((root / "summary.txt").exists())
        self.assertNotIn("精度: PASSED", stdout)
        self.assertIn("profile", stderr)
        self.assertEqual(json.loads((root / "one_click_result.json").read_text())["status"], "failed")

    def test_real_exporter_failure_keeps_raw_and_logs_without_summary(self):
        rc, root, stdout, _ = self.execute(export_failure=True)
        self.assertEqual(rc, 1)
        self.assertEqual(len(self.export_calls), 1)
        report = json.loads((root / "profile/result.json").read_text())
        self.assertEqual(report["accuracy"]["status"], "passed")
        self.assertIsNone(report["performance"])
        self.assertTrue(any("export failed (17)" in error for error in report["errors"]))
        manifest = json.loads((root / "profile/analysis/export_manifest.json").read_text())
        self.assertEqual((manifest["status"], manifest["returncode"]), ("failed", 17))
        self.assertIn("synthetic export failure", (root / "profile/analysis/export_stderr.log").read_text())
        self.assertFalse((root / "summary.txt").exists())
        self.assertNotIn("精度: PASSED", stdout)


if __name__ == "__main__":
    unittest.main()
