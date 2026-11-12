"""Four-node one-command workflow with actual export/parser/acceptance; transport and device are synthetic."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import dec_enc_profile as common, eight_head_profile as profile, host, profiling, run_profile
from tests import test_host as fixtures
from tests.test_profiling import add_profile_result


class EightHeadProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HostTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.resolve()
        self.nodes = list(profile.EXPECTED_NODES)
        audit = self.fixture.build / "atc/om_audit.json"
        audit.parent.mkdir()
        audit.write_text(json.dumps({"om_sha256": self.fixture.binding["om_sha256"],
                                    "graph": {"kernels": [{"name": name} for name in self.nodes]}}))
        self.fixture.binding["om_audit_sha256"] = host.sha256(audit)
        self.fixture.binding_path.write_text(json.dumps(self.fixture.binding))
        self.toolkit = self.root / "sdk"
        self.toolkit.mkdir()
        (self.toolkit / "set_env.sh").write_text("export TEST_SELECTED_SDK=yes\n")
        self.pto = self.root / "pto"
        (self.pto / "include/pto").mkdir(parents=True)
        (self.pto / "include/pto/pto-inst.hpp").write_text("// synthetic headers\n")
        self.parser = self.toolkit / "analysis/msprof/msprof.py"
        self.parser.parent.mkdir(parents=True)
        self.provider = self.root / "gym/provider.py"
        self.provider.parent.mkdir()
        self.provider.write_text("# synthetic provider path; compilation is mocked\n")
        self.config_path = self.root / "profile.json"
        self.raw = {"host": "synthetic1952", "toolkit_root": "sdk", "pto_root": "pto", "provider": "gym/provider.py",
                    "pypto_root": "pypto", "gym_root": "gym", "msprof_py": "sdk/analysis/msprof/msprof.py",
                    "output_root": "runs", "warmup": 1, "repeat": 2, "timeout": 30}
        self.config_path.write_text(json.dumps(self.raw))

    def exporter(self, fail=False):
        rows = []
        for execution in range(5):
            for node_index, name in enumerate(self.nodes):
                row = {"Op Name": name, "Task Start Time(us)": execution * 100 + node_index * 10,
                       "Task Duration(us)": 900000 + execution if execution < 3 else 10 + node_index + execution - 3}
                for index, (_, key) in enumerate(common.PIPELINES):
                    column = {"cube": "Cube_Exe_Time(us)", "vector": "Vec_Time(us)",
                              "mte1": "MTE1_Exe_Time(us)", "mte2": "MTE2_Exe_Time(us)",
                              "mte3": "MTE3_Exe_Time(us)", "fixpipe": "Fixpipe_Exe_Time(us)",
                              "scalar": "Scalar_Exe_Time(us)"}[key]
                    row[column] = index + execution if execution >= 3 else 900000
                rows.append(row)
        self.parser.write_text(
            "import csv,json,sys\nfrom pathlib import Path\n"
            "assert sys.argv[1:4] == ['export','summary','-dir']\n"
            f"if {fail!r}:\n print('synthetic export failure');sys.exit(19)\n"
            "output=Path(sys.argv[4])/'mindstudio_profiler_output'\noutput.mkdir()\n"
            f"rows=json.loads({json.dumps(rows)!r})\n"
            "with (output/'op_summary_fixture.csv').open('w',newline='') as stream:\n"
            " writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)\n")

    def execute(self, bad_final=False, export_failure=False, build_failure=False):
        self.exporter(export_failure)
        config = profile.load_config(self.config_path)
        add_profile_result(self.fixture, bad_final=bad_final)
        self.commands, self.transport_calls, self.export_calls = [], [], []
        real_run = subprocess.run

        def transport(command, **kwargs):
            if command[0] not in ("ssh", "scp"):
                self.export_calls.append(command)
                return real_run(command, **kwargs)
            self.transport_calls.append(command)
            if command[0] == "scp" and command[2].startswith(config["host"] + ":"):
                destination = Path(command[3])
                if command[2].endswith("/results"):
                    shutil.copytree(self.fixture.results, destination)
                else:
                    destination.mkdir()
                    (destination / "PROF_fixture").mkdir()
                    (destination / "PROF_fixture/raw.bin").write_bytes(b"synthetic raw transport, not device data")
                    state = json.loads((destination.parent / "profile_state.json").read_text())
                    capture = {"schema_version": 1, "run_id": state["run_id"],
                               "runner_sha256": state["runner_sha256"], "environment_rc": 0,
                               "runner_rc": 0, "profiler_rc": 0, "collection_rc": 0,
                               "prof": "PROF_fixture", "prof_directories": ["PROF_fixture"]}
                    (destination / "capture.json").write_text(json.dumps(capture))
            return SimpleNamespace(returncode=0, stdout="synthetic transport", stderr="")

        def stage(command, log, **kwargs):
            self.commands.append(command)
            if command[0] == "ssh":
                log.write_text("synthetic board preflight\n")
                return
            arguments = command[command.index("-m") + 2:]
            output = Path(arguments[arguments.index("--output") + 1])
            if arguments[0] == "build":
                self.assertEqual(arguments[arguments.index("--provider") + 1], str(self.provider))
                self.assertEqual(arguments[arguments.index("--block-dim") + 1], "8")
                self.assertNotIn("--build-dir", arguments)
                if build_failure:
                    raise RuntimeError("synthetic compile failure")
                shutil.copytree(self.fixture.build, output)
                (output / "build_result.json").write_text(json.dumps({"status": "bundle-built"}))
                log.write_text("synthetic build boundary\n")
            else:
                self.assertEqual(arguments[0], "profile")
                with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
                    rc = run_profile.main(arguments)
                if rc:
                    raise RuntimeError("profile failed; see " + str(log))

        with patch.object(common, "_run", side_effect=stage), \
                patch.object(profiling.subprocess, "run", side_effect=transport), \
                redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            rc = profile.execute(config)
        output = next(config["output_root"].iterdir())
        return rc, output, stdout.getvalue(), stderr.getvalue()

    def test_optional_cmake_and_relative_paths_and_explicit_provider(self):
        self.exporter()
        config = profile.load_config(self.config_path)
        self.assertIsNone(config["build_dir"])
        self.assertEqual(config["provider"], self.provider)
        self.assertEqual(config["toolkit_root"], self.toolkit)
        self.assertEqual(config["msprof_py"], self.parser)

    def test_default_gym_provider_and_config_validation(self):
        self.exporter()
        default_provider = self.root / "gym/ops/pro/experimental/eight_head_mlp_5102/provider.py"
        default_provider.parent.mkdir(parents=True)
        default_provider.write_text("# fixture\n")
        self.config_path.write_text(json.dumps({**self.raw, "provider": None}))
        self.assertEqual(profile.load_config(self.config_path)["provider"], default_provider)
        for field in ({"block_dim": 0}, {"block_dim": True}, {"provider": "/missing.py"},
                      {"password": "not a supported config field"}, {"build_dir": "missing-cmake"}):
            self.config_path.write_text(json.dumps({**self.raw, **field}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile.load_config(self.config_path)

    def test_init_preserves_existing_config_and_block_override(self):
        new = self.root / "local.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(profile.main(["--init", "--config", str(new)]), 0)
            original = new.read_bytes()
            self.assertEqual(profile.main(["--init", "--config", str(new)]), 1)
        self.assertEqual(new.read_bytes(), original)
        self.exporter()
        with patch.object(profile, "execute", return_value=0) as execute:
            self.assertEqual(profile.main(["--config", str(self.config_path), "--block-dim", "4"]), 0)
        self.assertEqual(execute.call_args[0][0]["block_dim"], 4)

    def test_actual_four_node_export_parse_acceptance_and_terminal_summary(self):
        rc, output, stdout, _ = self.execute()
        self.assertEqual(rc, 0)
        self.assertEqual([c[0] for c in self.transport_calls], ["ssh", "scp", "ssh", "scp", "scp"])
        self.assertEqual(len(self.export_calls), 1)
        report = json.loads((output / "profile/result.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["performance"]["expected_nodes"], self.nodes)
        for node_index, name in enumerate(self.nodes):
            self.assertIn("节点: " + name, stdout)
            node = report["performance"]["nodes"][name]
            self.assertEqual(node["measured_us"], [10 + node_index, 11 + node_index])
            for key in ("cube", "vector", "mte1", "mte2", "mte3", "fixpipe", "scalar"):
                self.assertEqual(node["pipeline_us"][key]["status"], "available")
        self.assertIn("整个 OM 同步主机调用", stdout)
        self.assertIn("设备任务累计（非端到端延迟）", stdout)
        self.assertNotIn("90000", (output / "summary.txt").read_text())
        self.assertEqual(json.loads((output / "one_click_result.json").read_text())["status"], "passed")
        self.assertTrue((output / "profile/raw/PROF_fixture/raw.bin").is_file())

    def test_bad_final_accuracy_prevents_export_and_success_summary(self):
        rc, output, stdout, _ = self.execute(bad_final=True)
        self.assertEqual(rc, 1)
        self.assertEqual(self.export_calls, [])
        self.assertNotIn("精度: PASSED", stdout)
        self.assertFalse((output / "summary.txt").exists())
        self.assertTrue((output / "profile/board/profile_actual.bin").is_file())

    def test_export_failure_keeps_raw_and_returns_nonzero(self):
        rc, output, stdout, _ = self.execute(export_failure=True)
        self.assertEqual(rc, 1)
        self.assertFalse((output / "summary.txt").exists())
        self.assertNotIn("精度: PASSED", stdout)
        self.assertTrue((output / "profile/raw/PROF_fixture/raw.bin").is_file())

    def test_compile_failure_stops_upload(self):
        rc, output, _, _ = self.execute(build_failure=True)
        self.assertEqual(rc, 1)
        self.assertEqual(self.transport_calls, [])
        state = json.loads((output / "one_click_result.json").read_text())
        self.assertEqual((state["status"], state["stage"]), ("failed", "build"))


if __name__ == "__main__":
    unittest.main()
