"""Timing protocol tests use structural CSV fixtures, never measured performance."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from pypto_1952.profiling_export import (
    export_and_parse,
    parse_profile_csv,
)

NODES = ["gate", "up", "activation", "down"]
COLUMNS = ["Op Name", "Task Start Time(us)", "Task Duration(us)"]


def profile_rows(nodes=NODES, warmup=1, repeat=2):
    rows = []
    for execution in range(2 + warmup + repeat):
        for index, name in enumerate(nodes):
            if execution < 2:
                duration = 1000
            elif execution < 2 + warmup:
                duration = 500
            else:
                duration = (index + 1) * 10 + (execution - 2 - warmup) * 2
            rows.append(
                {
                    "Op Name": name,
                    "Task Start Time(us)": execution * 100000 + index * 100,
                    "Task Duration(us)": duration,
                }
            )
    return rows


class ProfilingExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / "op_summary_0.csv"

    def write_csv(self, rows, columns=COLUMNS):
        with self.csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_four_nodes_exclude_prefix_and_warmup_independently(self):
        self.write_csv(list(reversed(profile_rows())))
        report = parse_profile_csv(self.csv, NODES, 1, 2, host_wall_seconds=[0.001, 0.002])
        self.assertEqual(report["execution_count_per_node"], 5)
        for index, name in enumerate(NODES):
            node = report["nodes"][name]
            self.assertEqual(node["correctness_prefix_us"], [1000, 1000])
            self.assertEqual(node["warmup_us"], [500])
            self.assertEqual(node["measured_us"], [(index + 1) * 10, (index + 1) * 10 + 2])
        self.assertEqual(report["device_task_sum_us"]["samples"], [100, 108])
        self.assertIn("not_end_to_end", report["device_task_sum_us"]["scope"])
        self.assertEqual(report["host_wall_seconds"]["units"], "seconds")
        self.assertEqual(report["host_wall_seconds"]["samples"], [0.001, 0.002])
        self.assertAlmostEqual(report["host_wall_seconds"]["statistics"]["mean"], 0.0015)

    def test_single_node_and_zero_warmup_preserve_all_repeat_samples(self):
        self.write_csv(profile_rows(["native_pro"], warmup=0, repeat=3))
        report = parse_profile_csv(self.csv, ["native_pro"], 0, 3)
        self.assertEqual(report["nodes"]["native_pro"]["measured_us"], [10, 12, 14])
        self.assertEqual(report["device_task_sum_us"]["samples"], [10, 12, 14])

    def test_official_task_duration_takes_precedence_over_pmu_time(self):
        rows = profile_rows()
        for row in rows:
            row["total_exe_time(us)"] = 999999
        self.write_csv(rows, COLUMNS + ["total_exe_time(us)"])
        report = parse_profile_csv(self.csv, NODES, 1, 2)
        self.assertEqual(report["duration_column"], "Task Duration(us)")
        self.assertEqual(report["nodes"]["gate"]["measured_us"], [10, 12])

    def test_known_mc62_total_exe_time_column_is_supported(self):
        rows = profile_rows()
        for row in rows:
            row["total_exe_time(us)"] = row.pop("Task Duration(us)")
        self.write_csv(rows, COLUMNS[:2] + ["total_exe_time(us)"])
        self.assertEqual(parse_profile_csv(self.csv, NODES, 1, 2)["duration_column"], "total_exe_time(us)")

    def test_optional_pipeline_counters_keep_only_repeat_and_preserve_missing_values(self):
        rows = profile_rows()
        pipeline = ["total_exe_time(us)", "aic_mac_time(us)", "vec_exe_time(us)", "scalar_exe_time(us)"]
        for index, row in enumerate(rows):
            execution = index // len(NODES)
            row.update(
                {
                    pipeline[0]: 999 if execution < 3 else 20 + execution,
                    pipeline[1]: 999 if execution < 3 else 2 + execution,
                    pipeline[2]: 0,
                    pipeline[3]: "N/A",
                }
            )
        self.write_csv(rows, COLUMNS + pipeline)
        report = parse_profile_csv(self.csv, NODES, 1, 2)
        values = report["nodes"]["gate"]["pipeline_us"]
        self.assertEqual(values["hardware_total"]["samples"], [23, 24])
        self.assertEqual(values["hardware_total"]["statistics"]["median"], 23.5)
        self.assertEqual(values["cube"]["samples"], [5, 6])
        self.assertEqual(values["cube"]["column"], "aic_mac_time(us)")
        self.assertEqual(values["vector"]["statistics"]["median"], 0)
        for metric in ("scalar", "mte1"):
            self.assertEqual(values[metric]["status"], "not_available")
            self.assertEqual(values[metric]["samples"], [None, None])
            self.assertIsNone(values[metric]["statistics"])
        self.assertEqual(report["nodes"]["gate"]["measured_us"], [10, 12])

    def test_partially_missing_pipeline_metric_has_no_invented_statistics(self):
        rows = profile_rows()
        for row in rows:
            row["mte2_exe_time(us)"] = 7
        rows[-4]["mte2_exe_time(us)"] = "-"
        self.write_csv(rows, COLUMNS + ["mte2_exe_time(us)"])
        value = parse_profile_csv(self.csv, NODES, 1, 2)["nodes"]["gate"]["pipeline_us"]["mte2"]
        self.assertEqual(value["samples"], [7, None])
        self.assertEqual(value["status"], "not_available")
        self.assertIsNone(value["statistics"])

    def test_all_seven_pipeline_times_support_sdk_and_formatted_headers(self):
        aliases = {
            "cube": "Cube_Exe_Time (us)",
            "vector": "Vec_Time (us)",
            "mte1": "MTE1 Exe Time (us)",
            "mte2": "aic_mte2_time(us)",
            "mte3": "mte3_time(us)",
            "fixpipe": "fixpipe_exe_time(us)",
            "scalar": "aic_scalar_time(us)",
        }
        rows = profile_rows()
        for index, row in enumerate(rows):
            execution = index // len(NODES)
            for metric_index, column in enumerate(aliases.values()):
                row[column] = 9999 if execution < 3 else metric_index + execution
        self.write_csv(rows, COLUMNS + list(aliases.values()))
        report = parse_profile_csv(self.csv, NODES, 1, 2)
        for metric_index, (metric, column) in enumerate(aliases.items()):
            value = report["nodes"]["gate"]["pipeline_us"][metric]
            self.assertEqual(value["column"], column)
            self.assertEqual(value["status"], "available")
            self.assertEqual(value["samples"], [metric_index + 3, metric_index + 4])
            self.assertEqual(value["statistics"]["median"], metric_index + 3.5)

    def test_pipeline_cycles_ratios_other_units_and_other_core_are_not_us(self):
        columns = [
            "mac_exe_time(cycles)",
            "vec_exe_ratio",
            "mte1_exe_time(ns)",
            "mte2_exe_time(ms)",
            "mte3_cycles",
            "fixpipe_ratio(%)",
            "scalar_exe_time",
            "aiv_vec_time(us)",
            "aiv_mte2_time(us)",
            "aiv_mte3_time(us)",
        ]
        rows = profile_rows()
        for row in rows:
            row.update(dict.fromkeys(columns, 12345))
        self.write_csv(rows, COLUMNS + columns)
        values = parse_profile_csv(self.csv, NODES, 1, 2)["nodes"]["gate"]["pipeline_us"]
        for metric in ("cube", "vector", "mte1", "mte2", "mte3", "fixpipe", "scalar"):
            with self.subTest(metric=metric):
                self.assertEqual(values[metric]["status"], "not_available")
                self.assertIsNone(values[metric]["column"])
                self.assertEqual(values[metric]["samples"], [None, None])
                self.assertIsNone(values[metric]["statistics"])

    def test_pipeline_alias_precedence_and_ambiguous_spelling(self):
        rows = profile_rows()
        for row in rows:
            row.update({"mac_exe_time(us)": 1, "cube_exe_time(us)": 9999})
        self.write_csv(rows, COLUMNS + ["mac_exe_time(us)", "cube_exe_time(us)"])
        value = parse_profile_csv(self.csv, NODES, 1, 2)["nodes"]["gate"]["pipeline_us"]["cube"]
        self.assertEqual(value["column"], "mac_exe_time(us)")
        self.assertEqual(value["samples"], [1, 1])
        for row in rows:
            row["MAC Exe Time (us)"] = 2
        self.write_csv(rows, COLUMNS + ["mac_exe_time(us)", "cube_exe_time(us)", "MAC Exe Time (us)"])
        with self.assertRaisesRegex(ValueError, "Ambiguous pipeline duration"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def test_invalid_measured_pipeline_counter_is_not_silently_accepted(self):
        rows = profile_rows()
        for row in rows:
            row["fixpipe_time(us)"] = "N/A"
        rows[-1]["fixpipe_time(us)"] = -1
        self.write_csv(rows, COLUMNS + ["fixpipe_time(us)"])
        with self.assertRaisesRegex(ValueError, "pipeline duration"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def test_generic_or_host_duration_columns_cannot_be_device_time(self):
        for column in ("Duration(us)", "Host Duration(us)", "Task Duration(ms)"):
            rows = profile_rows()
            for row in rows:
                row[column] = row.pop("Task Duration(us)")
            self.write_csv(rows, COLUMNS[:2] + [column])
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "explicit device duration"):
                parse_profile_csv(self.csv, NODES, 1, 2)

    def test_missing_and_extra_kernel_counts_fail(self):
        for extra in (False, True):
            rows = profile_rows()
            if extra:
                rows.append({"Op Name": "gate", "Task Start Time(us)": 9999999, "Task Duration(us)": 1})
            else:
                rows.pop()
            self.write_csv(rows)
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "kernel rows"):
                parse_profile_csv(self.csv, NODES, 1, 2)

    def test_native_entry_name_cannot_replace_three_distinct_gemm_nodes(self):
        rows = profile_rows()
        rows[0]["Op Name"] = "gemm"
        self.write_csv(rows)
        with self.assertRaisesRegex(ValueError, "Unexpected node"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def test_missing_timestamp_and_duplicate_timestamps_fail(self):
        rows = profile_rows()
        for row in rows:
            row.pop("Task Start Time(us)")
        self.write_csv(rows, [COLUMNS[0], COLUMNS[2]])
        with self.assertRaisesRegex(ValueError, "Task Start Time"):
            parse_profile_csv(self.csv, NODES, 1, 2)
        rows = profile_rows()
        rows[4]["Task Start Time(us)"] = rows[0]["Task Start Time(us)"]
        self.write_csv(rows)
        with self.assertRaisesRegex(ValueError, "duplicate task timestamps"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def test_empty_or_nonfinite_device_timings_fail(self):
        self.write_csv([])
        with self.assertRaisesRegex(ValueError, "no kernel rows"):
            parse_profile_csv(self.csv, NODES, 1, 2)
        for value in ("", "N/A", "nan", "inf", "0", "-1"):
            rows = profile_rows()
            rows[-1]["Task Duration(us)"] = value
            self.write_csv(rows)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "device duration"):
                parse_profile_csv(self.csv, NODES, 1, 2)

    def test_invalid_protocol_and_host_wall_counts_fail(self):
        self.write_csv(profile_rows())
        for names, warmup, repeat in ((["gate", "gate"], 1, 2), (NODES, -1, 2), (NODES, 1, 0)):
            with self.assertRaises(ValueError):
                parse_profile_csv(self.csv, names, warmup, repeat)
        with self.assertRaisesRegex(ValueError, "exactly repeat"):
            parse_profile_csv(self.csv, NODES, 1, 2, host_wall_seconds=[1])

    def test_duplicate_csv_headers_and_overlapping_execution_cycles_fail(self):
        self.csv.write_text("Op Name,Task Start Time(us),Task Duration(us),Task Duration(us)\n")
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            parse_profile_csv(self.csv, NODES, 1, 2)
        rows = profile_rows()
        for index, row in enumerate(rows):
            if row["Op Name"] == "down":
                row["Task Start Time(us)"] = index + 1
        self.write_csv(rows)
        with self.assertRaisesRegex(ValueError, "sequential model executions"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def test_malformed_csv_header_fails_as_a_protocol_error(self):
        self.csv.write_text('"unterminated CSV header')
        with self.assertRaisesRegex(ValueError, "Malformed op_summary CSV header"):
            parse_profile_csv(self.csv, NODES, 1, 2)

    def fake_exporter(self, *, fail=False, summaries=1):
        """An explicit fake tool for process/log preservation tests only."""
        suite = self.root / "suite"
        script = suite / "analysis/msprof/msprof.py"
        script.parent.mkdir(parents=True)
        (suite / "fixture_rows.json").write_text(json.dumps(profile_rows()))
        script.write_text(
            "import csv,json,sys\nfrom pathlib import Path\n"
            "print('STRUCTURAL FAKE EXPORTER; no profiling measurements')\n"
            "print('preserved stderr',file=sys.stderr)\n"
            "assert sys.argv[1:4]==['export','summary','-dir']\n"
            "assert Path.cwd()==Path(__file__).resolve().parents[2]\n"
            f"sys.exit(7) if {fail!r} else None\n"
            "rows=json.loads(Path('fixture_rows.json').read_text())\n"
            "out=Path(sys.argv[4])/'mindstudio_profiler_output';out.mkdir()\n"
            f"for index in range({summaries}):\n"
            " with (out/('op_summary_%d.csv'%index)).open('w',newline='') as stream:\n"
            "  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)\n"
        )
        prof = self.root / "PROF_fixture"
        raw = prof / "device_0/data/raw.bin"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"STRUCTURAL TEST INPUT; not measured profiling data")
        stale = prof / "mindstudio_profiler_output/op_summary_old.csv"
        stale.parent.mkdir()
        stale.write_text("OLD EXPORT MUST NOT ENTER RESULTS\n")
        return prof, script, raw, stale

    def test_export_uses_real_cli_shape_and_retains_source_raw_old_exports_and_logs(self):
        prof, script, raw, stale = self.fake_exporter()
        output = self.root / "export"
        report = export_and_parse(prof, script, output, NODES, 1, 2)
        self.assertEqual(report["device_task_sum_us"]["samples"], [100, 108])
        self.assertEqual(raw.read_bytes(), (output / prof.name / "device_0/data/raw.bin").read_bytes())
        self.assertEqual(stale.read_text(), (output / "previous_exports/op_summary_old.csv").read_text())
        self.assertIn("FAKE EXPORTER", (output / "export_stdout.log").read_text())
        self.assertIn("preserved stderr", (output / "export_stderr.log").read_text())
        self.assertEqual(json.loads((output / "export_manifest.json").read_text())["status"], "passed")
        self.assertTrue((output / "profiling_summary.json").is_file())
        with (output / "node_statistics.csv").open(newline="") as stream:
            statistics_rows = list(csv.DictReader(stream))
        task = next(row for row in statistics_rows if row["node"] == "gate" and row["metric"] == "task_duration")
        self.assertEqual(float(task["median"]), 11)
        unavailable = next(row for row in statistics_rows if row["node"] == "gate" and row["metric"] == "cube")
        self.assertEqual(unavailable["status"], "not_available")
        self.assertEqual(unavailable["median"], "")

    def test_failed_export_preserves_logs_and_cannot_publish_timings(self):
        prof, script, _, _ = self.fake_exporter(fail=True)
        output = self.root / "export"
        with self.assertRaisesRegex(RuntimeError, "export failed"):
            export_and_parse(prof, script, output, NODES, 1, 2)
        self.assertTrue((output / "export_stderr.log").exists())
        manifest = json.loads((output / "export_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["returncode"], 7)
        self.assertFalse((output / "profiling_summary.json").exists())

    def test_statistics_csv_preserves_all_seven_pipeline_metrics_and_na(self):
        prof, script, _, _ = self.fake_exporter()
        rows = profile_rows()
        pipeline = {
            "cube": "mac_exe_time(us)",
            "vector": "vec_exe_time(us)",
            "mte1": "mte1_time(us)",
            "mte2": "mte2_time(us)",
            "mte3": "mte3_time(us)",
            "fixpipe": "fixpipe_exe_time(us)",
            "scalar": "scalar_time(us)",
        }
        for row in rows:
            row.update({column: index for index, column in enumerate(pipeline.values())})
            row[pipeline["scalar"]] = "N/A"
        (script.parents[2] / "fixture_rows.json").write_text(json.dumps(rows))
        output = self.root / "export"
        export_and_parse(prof, script, output, NODES, 1, 2)
        with (output / "node_statistics.csv").open(newline="") as stream:
            statistics_rows = list(csv.DictReader(stream))
        for index, (metric, column) in enumerate(pipeline.items()):
            record = next(row for row in statistics_rows if row["node"] == "gate" and row["metric"] == metric)
            self.assertEqual(record["source_column"], column)
            self.assertEqual(record["units"], "us")
            if metric == "scalar":
                self.assertEqual(record["status"], "not_available")
                self.assertTrue(all(record[field] == "" for field in ("min", "median", "mean", "max")))
            else:
                self.assertEqual(record["status"], "available")
                self.assertEqual(float(record["median"]), index)

    def test_empty_raw_capture_cannot_be_exported_as_success(self):
        prof, script, raw, _ = self.fake_exporter()
        raw.unlink()
        with self.assertRaisesRegex(ValueError, "empty or contains only old"):
            export_and_parse(prof, script, self.root / "export", NODES, 1, 2)
        self.assertFalse((self.root / "export/profiling_summary.json").exists())

    def test_missing_or_multiple_summary_files_cannot_pick_a_stale_export(self):
        for count in (0, 2):
            with tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary)
                prof, script, _, _ = self.fake_exporter(summaries=count)
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "exactly one"):
                    export_and_parse(prof, script, self.root / "export", NODES, 1, 2)

    def test_stale_summary_in_an_unexpected_subdirectory_cannot_substitute_for_export(self):
        prof, script, _, _ = self.fake_exporter(summaries=0)
        old = prof / "old-cache/op_summary_stale.csv"
        old.parent.mkdir()
        with old.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(profile_rows())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            export_and_parse(prof, script, self.root / "export", NODES, 1, 2)
        self.assertTrue(old.exists())
        self.assertTrue((self.root / "export/previous_exports/other_summaries/old-cache/op_summary_stale.csv").exists())


if __name__ == "__main__":
    unittest.main()
