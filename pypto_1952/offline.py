"""Compile/package on A, execute the runtime archive on B, accept returned results on A. No SSH."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tarfile
from types import SimpleNamespace

from . import host, offline_run

HERE = Path(__file__).resolve().parent
RUN_SH = '''#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "--runtime-env" ]]; then
    if [[ $# -lt 2 ]]; then echo "--runtime-env requires a board environment script" >&2; exit 2; fi
    if source "$2"; then :; else echo "Board runtime environment failed: $2" >&2; exit 1; fi
    shift 2
fi
set -eo pipefail
exec "${PYTHON:-python3}" -B "$script_dir/offline_run.py" "$@"
'''


def package(build, output):
    root, case, binding, _ = host.validate_build(build)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    shutil.copytree(root / "bundle", runtime / "bundle")
    shutil.copy2(HERE / "offline_run.py", runtime / "offline_run.py")
    (runtime / "run.sh").write_text(RUN_SH)
    (runtime / "run.sh").chmod(0o755)
    build_report = json.loads((root / "build_result.json").read_text())
    if build_report.get("status") != "bundle-built":
        raise ValueError("Package only a completed static OM build")
    manifest = {"schema_version": 1, "kind": "pypto_1952_offline_runtime", "operator": case["operator"],
                "target_soc": build_report["target_soc"], "om_sha256": binding["om_sha256"],
                "request_sha256": host.sha256(root / "bundle/execution.json"),
                "case_sha256": binding["case_sha256"], "minimum_board_python": "3.7.9",
                "runtime_dependencies": "AscendCL runtime and compatible driver/firmware",
                "device_execution": "not_run",
                "files": {p.relative_to(runtime).as_posix(): host.sha256(p)
                          for p in sorted(runtime.rglob("*")) if p.is_file()}}
    (runtime / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    offline_run.verify(runtime)
    archive = output / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(runtime, arcname="runtime")
    report = {"schema_version": 1, "status": "offline-package-built", "build": str(root),
              "compiler_python": sys.version, "compiler_machine": platform.machine(),
              "build_report": build_report, "om_sha256": binding["om_sha256"],
              "case_sha256": binding["case_sha256"], "request_sha256": manifest["request_sha256"],
              "runtime_manifest_sha256": host.sha256(runtime / "runtime.json"),
              "archive": {"file": archive.name, "sha256": host.sha256(archive), "bytes": archive.stat().st_size},
              "device_execution": "not_run"}
    (output / "package_result.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(report["archive"]["sha256"] + "  runtime.tar.gz\n")
    print(json.dumps(report, indent=2))
    return report


def _unpack_results(archive, destination):
    allowed = {"board_run.json", "runner.log", "results/execution.json", "results/actual.bin",
               "results/actual_second.bin", "results/profile_actual.bin"}
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unexpected, duplicate or unsafe result archive member: " + member.name)
            seen.add(member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as stream:
                shutil.copyfileobj(source, stream)
    if not {"board_run.json", "runner.log"} <= seen:
        raise ValueError("Returned archive lacks board execution evidence")


def accept(build, archive, output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "failed", "precision_checked_on": "compiler_host"}
    try:
        root, _, binding, _ = host.validate_build(build)
        returned = output / "returned"
        returned.mkdir()
        _unpack_results(archive, returned)
        board = json.loads((returned / "board_run.json").read_text())
        if (board.get("schema_version") != 1 or board.get("kind") != "pypto_1952_offline_result"
                or board.get("status") != "executed" or board.get("runner_rc") != 0
                or board.get("om_sha256") != binding["om_sha256"]
                or board.get("request_sha256") != host.sha256(root / "bundle/execution.json")):
            raise ValueError("Board failed or executed a different OM/request; see returned/runner.log")
        report = host.evaluate(root, returned / "results")
        if board["repeat"]:
            from .profiling import validate_profile_execution

            execution, comparison = validate_profile_execution(root, returned / "results", SimpleNamespace(**board))
            report["profile_final_comparison"] = comparison
            if not comparison["passed"]:
                report["status"] = "failed"
            if report["status"] == "passed":
                values = [v * 1e6 for v in execution["host_wall_seconds"]]
                report["whole_om_host_wall_us"] = {
                    "scope": "synchronous_aclmdlExecute_host_wall_not_device_kernel_time",
                    "count": len(values), "min": min(values), "median": statistics.median(values),
                    "mean": statistics.mean(values), "max": max(values), "stddev": statistics.pstdev(values)}
        report["results_archive_sha256"] = host.sha256(archive)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError) as error:
        report.update(status="failed", error=str(error))
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "build":
        # Reuse the existing generic provider CLI and its exact options; no second compiler implementation.
        from .run_profile import main as build_main

        if "--help" in argv or "-h" in argv:
            return build_main(argv)
        output_parser = argparse.ArgumentParser(add_help=False)
        output_parser.add_argument("--output", type=Path, required=True)
        options, _ = output_parser.parse_known_args(argv[1:])
        rc = build_main(argv)
        if rc:
            return rc
        try:
            package(options.output, options.output / "offline")
            return 0
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            print("Offline package failed: " + str(error), file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="build: existing build options, plus automatic offline packaging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Native device compile + ATC + case/golden + offline archive")
    pack = commands.add_parser("package", help="Create an offline archive from an existing audited build")
    pack.add_argument("--build", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("accept", help="Accept the returned board archive on the compiler host")
    compare.add_argument("--build", type=Path, required=True)
    compare.add_argument("--results-archive", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            package(args.build, args.output)
            return 0
        return accept(args.build, args.results_archive, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Offline workflow failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
