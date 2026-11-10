"""Compiler-side msprof export and strict native-node device timing summaries."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

PIPELINE_COLUMNS = {
    "hardware_total": ("total_exe_time(us)", "aicore_time(us)"),
    "cube": ("mac_exe_time(us)", "cube_exe_time(us)", "aic_mac_time(us)", "mac_time(us)"),
    "vector": ("vec_exe_time(us)", "vec_time(us)", "aic_vec_time(us)"),
    "scalar": ("scalar_exe_time(us)", "aic_scalar_time(us)", "scalar_time(us)"),
    "mte1": ("mte1_exe_time(us)", "aic_mte1_time(us)", "mte1_time(us)"),
    "mte2": ("mte2_exe_time(us)", "aic_mte2_time(us)", "mte2_time(us)"),
    "mte3": ("mte3_exe_time(us)", "mte3_time(us)"),
    "fixpipe": ("fixpipe_time(us)", "fixpipe_exe_time(us)", "aic_fixpipe_time(us)"),
}


def _pipeline_column(columns, aliases):
    # CANN exports also use e.g. "Cube_Exe_Time (us)". Normalize spelling
    # separators only: parentheses and units remain part of the exact key,
    # so cycles, ratios, ns and ms never become microsecond measurements.
    def key(column):
        return re.sub(r"[\s_]+", "", column).lower()

    for alias in aliases:
        matches = [column for column in columns if key(column) == key(alias)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous pipeline duration columns: {matches}")
        if matches:
            return matches[0]
    return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, label, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def _statistics(samples):
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "max": max(samples),
    }


def _pipeline_metric(rows, measured, column):
    samples = []
    for task in measured:
        value = rows[task["csv_row"] - 2].get(column) if column else None
        if value is None or str(value).strip().lower() in ("", "n/a", "na", "-"):
            samples.append(None)
        else:
            samples.append(_number(value, "pipeline duration " + column))
    complete = all(value is not None for value in samples)
    return {
        "status": "available" if complete else "not_available",
        "column": column,
        "samples": samples,
        "statistics": _statistics(samples) if complete else None,
    }


def _protocol(expected_nodes, warmup, repeat, prefix_executions):
    if (
        not isinstance(expected_nodes, (list, tuple))
        or not expected_nodes
        or any(not isinstance(name, str) or not name or name != name.strip() for name in expected_nodes)
        or len(set(expected_nodes)) != len(expected_nodes)
    ):
        raise ValueError("Require distinct native node names from the actual OM audit")
    if (
        type(warmup) is not int
        or warmup < 0
        or type(repeat) is not int
        or repeat < 1
        or type(prefix_executions) is not int
        or prefix_executions < 0
    ):
        raise ValueError("Require nonnegative integer warmup/prefix_executions and positive integer repeat")
    return prefix_executions + warmup + repeat


def parse_profile_csv(csv_path, expected_nodes, warmup, repeat, *, prefix_executions=2, host_wall_seconds=None):
    """Exclude correctness/warmup for each node; report only known device durations.

    Task Duration(us) is the official task elapsed time and takes precedence.
    total_exe_time(us) is the supported older MC62 export spelling. Generic
    Duration/Host Duration columns are deliberately not treated as device time.
    """
    execution_count = _protocol(expected_nodes, warmup, repeat, prefix_executions)
    csv_path = Path(csv_path).resolve(strict=True)
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        try:
            columns = reader.fieldnames
        except csv.Error as error:
            raise ValueError("Malformed op_summary CSV header") from error
        if not columns or any(not column for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("op_summary requires a nonempty CSV header without duplicate columns")
        duration = next((column for column in ("Task Duration(us)", "total_exe_time(us)") if column in columns), None)
        name_column = next(
            (column for column in ("Op Name", "OpName", "Node Name", "NodeName", "Name") if column in columns), None
        )
        timestamp = "Task Start Time(us)"
        if duration is None or name_column is None or timestamp not in columns:
            raise ValueError(
                "op_summary requires an explicit device duration, native node name and Task Start Time(us)"
            )
        try:
            rows = list(reader)
        except csv.Error as error:
            raise ValueError("Malformed op_summary CSV") from error
    if not rows:
        raise ValueError("op_summary contains no kernel rows")
    tasks = {name: [] for name in expected_nodes}
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"op_summary row {index} differs from the CSV header")
        name = row[name_column].strip()
        if name not in tasks:
            task_type = row.get("Task Type", row.get("TaskType"))
            raise ValueError(
                f"Unexpected node {name!r} (Task Type={task_type!r}) in op_summary; "
                "require the actual audited native nodes; no task types are silently ignored"
            )
        tasks[name].append(
            {
                "duration_us": _number(row[duration], "device duration", positive=True),
                "start_us": _number(row[timestamp], "device task timestamp"),
                "csv_row": index,
            }
        )
    for name, samples in tasks.items():
        if len(samples) != execution_count:
            raise ValueError(
                f"Native node {name!r} has {len(samples)} kernel rows; expected {execution_count} "
                "including correctness prefix, warmup and repeat"
            )
        samples.sort(key=lambda sample: sample["start_us"])
        if len({sample["start_us"] for sample in samples}) != len(samples):
            raise ValueError(f"Native node {name!r} has ambiguous duplicate task timestamps")
    for index in range(execution_count - 1):
        latest_start = max(samples[index]["start_us"] for samples in tasks.values())
        next_start = min(samples[index + 1]["start_us"] for samples in tasks.values())
        if latest_start >= next_start:
            raise ValueError("Native node timestamps do not describe distinct sequential model executions")
    selected = slice(prefix_executions + warmup, execution_count)
    pipeline_columns = {metric: _pipeline_column(columns, aliases) for metric, aliases in PIPELINE_COLUMNS.items()}
    nodes = {}
    for name, samples in tasks.items():
        measured = samples[selected]
        durations = [sample["duration_us"] for sample in measured]
        nodes[name] = {
            "execution_count": execution_count,
            "correctness_prefix_us": [sample["duration_us"] for sample in samples[:prefix_executions]],
            "warmup_us": [sample["duration_us"] for sample in samples[prefix_executions : prefix_executions + warmup]],
            "measured_us": durations,
            "statistics_us": _statistics(durations),
            "measured_task_start_us": [sample["start_us"] for sample in measured],
            "measured_csv_rows": [sample["csv_row"] for sample in measured],
            "pipeline_us": {
                metric: _pipeline_metric(rows, measured, column) for metric, column in pipeline_columns.items()
            },
        }
    totals = [
        _number(
            sum(nodes[name]["measured_us"][index] for name in expected_nodes),
            "summed native device duration",
            positive=True,
        )
        for index in range(repeat)
    ]
    host_wall = {"scope": "whole_OM_API_wall_time", "units": "seconds", "samples": None, "statistics": None}
    if host_wall_seconds is not None:
        if not isinstance(host_wall_seconds, (list, tuple)) or len(host_wall_seconds) != repeat:
            raise ValueError("Host wall time requires exactly repeat whole-OM API samples in seconds")
        values = [_number(value, "host wall seconds", positive=True) for value in host_wall_seconds]
        host_wall.update(samples=values, statistics=_statistics(values))
    return {
        "schema_version": 1,
        "source": "msprof_op_summary",
        "device_units": "us",
        "op_summary": str(csv_path),
        "op_summary_sha256": _sha256(csv_path),
        "duration_column": duration,
        "node_name_column": name_column,
        "timestamp_column": timestamp,
        "ordering": "device_task_timestamp_per_native_node",
        "expected_nodes": list(expected_nodes),
        "ignored_rows": [],
        "prefix_executions": prefix_executions,
        "warmup": warmup,
        "repeat": repeat,
        "execution_count_per_node": execution_count,
        "nodes": nodes,
        "pipeline_columns": pipeline_columns,
        "device_task_sum_us": {
            "scope": "sum_native_task_durations_not_end_to_end_latency",
            "samples": totals,
            "statistics": _statistics(totals),
        },
        "host_wall_seconds": host_wall,
    }


def _write_statistics_csv(report, path):
    columns = ["node", "metric", "scope", "units", "status", "source_column", "samples", "min", "median", "mean", "max"]
    records = []
    for name, node in report["nodes"].items():
        records.append(
            {
                "node": name,
                "metric": "task_duration",
                "scope": "native_device_task",
                "units": "us",
                "status": "available",
                "source_column": report["duration_column"],
                "samples": report["repeat"],
                **node["statistics_us"],
            }
        )
        for metric, value in node["pipeline_us"].items():
            records.append(
                {
                    "node": name,
                    "metric": metric,
                    "scope": "hardware_pipeline_counter",
                    "units": "us",
                    "status": value["status"],
                    "source_column": value["column"],
                    "samples": report["repeat"],
                    **(value["statistics"] or {}),
                }
            )
    for metric, key, units in (
        ("native_task_sum", "device_task_sum_us", "us"),
        ("host_wall", "host_wall_seconds", "seconds"),
    ):
        value = report[key]
        records.append(
            {
                "node": "__model__",
                "metric": metric,
                "scope": value["scope"],
                "units": units,
                "status": "available" if value["statistics"] is not None else "not_available",
                "samples": report["repeat"],
                **(value["statistics"] or {}),
            }
        )
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def export_and_parse(
    prof_dir,
    msprof_py,
    output_dir,
    expected_nodes,
    warmup,
    repeat,
    *,
    prefix_executions=2,
    host_wall_seconds=None,
    python=None,
    timeout=600,
):
    """Export a copy of recovered PROF data; retain raw data, all outputs and logs.

    Uses the verified CANN/TbeWorkTestSuit CLI:
    python analysis/msprof/msprof.py export summary -dir <PROF>.
    Both SDK profiler_tool and a TbeWorkTestSuit root equal msprof.py.parents[2].
    """
    _protocol(expected_nodes, warmup, repeat, prefix_executions)
    prof_dir, msprof_py = Path(prof_dir).resolve(strict=True), Path(msprof_py).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if not prof_dir.is_dir() or not msprof_py.is_file():
        raise ValueError("Require a recovered PROF directory and the actual msprof.py parser")
    files = list(prof_dir.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("Recovered PROF data must not contain symlinks")
    if not any(
        path.is_file()
        and path.stat().st_size
        and "mindstudio_profiler_output" not in path.parts
        and not (path.name.startswith("op_summary_") and path.suffix == ".csv")
        for path in files
    ):
        raise ValueError("Recovered PROF directory is empty or contains only old exported summaries")
    if output_dir == prof_dir or prof_dir in output_dir.parents:
        raise ValueError("Export output must be outside the recovered PROF source")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a fresh profiling export output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    working_prof = output_dir / prof_dir.name
    shutil.copytree(prof_dir, working_prof)
    old_exports = working_prof / "mindstudio_profiler_output"
    if old_exports.exists():
        shutil.move(str(old_exports), str(output_dir / "previous_exports"))
    for old_summary in working_prof.rglob("op_summary_*.csv"):
        backup = output_dir / "previous_exports/other_summaries" / old_summary.relative_to(working_prof)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_summary), str(backup))
    command = [str(python or sys.executable), str(msprof_py), "export", "summary", "-dir", str(working_prof)]
    manifest = {
        "schema_version": 1,
        "status": "failed",
        "source_prof": str(prof_dir),
        "working_prof": str(working_prof),
        "msprof_py": str(msprof_py),
        "msprof_py_sha256": _sha256(msprof_py),
        "command": command,
        "cwd": str(msprof_py.parents[2]),
        "prefix_executions": prefix_executions,
        "warmup": warmup,
        "repeat": repeat,
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        stdout_log, stderr_log = output_dir / "export_stdout.log", output_dir / "export_stderr.log"
        with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
            result = subprocess.run(
                command, cwd=msprof_py.parents[2], stdout=stdout, stderr=stderr, timeout=timeout, check=False
            )
        manifest["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"msprof summary export failed ({result.returncode}); see export logs")
        summaries = sorted(working_prof.rglob("op_summary_*.csv"))
        if len(summaries) != 1:
            raise ValueError(f"Require exactly one exported op_summary CSV, found {len(summaries)}")
        report = parse_profile_csv(
            summaries[0],
            expected_nodes,
            warmup,
            repeat,
            prefix_executions=prefix_executions,
            host_wall_seconds=host_wall_seconds,
        )
        statistics_path = output_dir / "node_statistics.csv"
        _write_statistics_csv(report, statistics_path)
        report["statistics_csv"] = {"path": str(statistics_path), "sha256": _sha256(statistics_path)}
        report_path = output_dir / "profiling_summary.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(status="passed", summary=str(report_path), summary_sha256=_sha256(report_path))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest["error"] = str(error)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return report
