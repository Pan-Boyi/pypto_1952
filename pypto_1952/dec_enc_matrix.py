"""Compiler-host matrix driver for native multicore DecEnc build/run/profile."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from .provider import load_provider


def cases(matrix, block_dim, rows, schedule="pipelined", gym_root=None):
    _, provider = load_provider(gym_root=gym_root)
    return provider.dec_enc_cases(matrix, block_dim, rows, schedule)


def command(args, config):
    name = (
        f"h{config['height']}_w{config['width']}_{config['dtype']}_b{config['block_dim']}_r{config['rows']}"
        f"_{args.dec_enc_schedule}"
    )
    task = args.root / name
    cmd = [sys.executable, "-m", "pypto_1952.run_profile", args.phase]
    if args.phase == "build":
        cmd += [
            "--gym-root",
            str(args.gym_root),
            "--op",
            "dec-enc",
            "--dec-enc-schedule",
            args.dec_enc_schedule,
            "--toolkit-root",
            str(args.toolkit_root),
            "--pto-root",
            str(args.pto_root),
            "--target-soc",
            args.target_soc,
            "--height",
            str(config["height"]),
            "--width",
            str(config["width"]),
            "--dtype",
            config["dtype"],
            "--block-dim",
            str(config["block_dim"]),
            "--rows-per-tile",
            str(config["rows"]),
            "--output",
            str(task / "build"),
        ]
    else:
        remote = args.remote_root.rstrip("/") + "/" + name + "/" + args.results_tag
        cmd += [
            "--build",
            str(task / "build"),
            "--host",
            args.host,
            "--remote-dir",
            remote,
            "--python",
            args.python,
            "--device",
            str(args.device),
            "--timeout",
            str(args.timeout),
            "--output",
            str(task / args.results_tag),
        ]
        if args.runtime_env:
            cmd += ["--runtime-env", args.runtime_env]
        if args.phase == "profile":
            cmd += [
                "--msprof",
                args.msprof,
                "--msprof-py",
                str(args.msprof_py),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
            ]
    return name, cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, default=Path(__file__).resolve().parents[2] / "pypto-gym")
    parser.add_argument("--phase", choices=("build", "run", "profile"), required=True)
    parser.add_argument("--matrix", choices=("smoke", "full"), default="smoke", help="4 boundary cases or all 32")
    parser.add_argument("--root", type=Path, required=True, help="Shared compiler-side matrix root across phases")
    parser.add_argument("--toolkit-root", type=Path)
    parser.add_argument("--pto-root", type=Path)
    parser.add_argument("--target-soc", default="MC62CM12AA")
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--rows-per-tile", type=int)
    parser.add_argument("--dec-enc-schedule", choices=("baseline", "optimized", "pipelined"), default="pipelined")
    parser.add_argument("--host")
    parser.add_argument("--remote-root")
    parser.add_argument("--results-tag", help="Fresh result subdirectory; defaults to the phase")
    parser.add_argument("--runtime-env")
    parser.add_argument("--python", default="python3", help="Board Python 3.7.9+")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--msprof", default="/var/msprof")
    parser.add_argument("--msprof-py", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args(argv)
    if args.phase == "build" and (not args.toolkit_root or not args.pto_root):
        parser.error("build requires --toolkit-root and --pto-root")
    if args.phase != "build" and (not args.host or not args.remote_root):
        parser.error("run/profile require --host and --remote-root")
    if args.phase == "profile" and not args.msprof_py:
        parser.error("profile requires the compiler-side --msprof-py")
    args.results_tag = args.results_tag or args.phase
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.results_tag) or (args.phase != "build" and args.results_tag == "build"):
        parser.error("Use a simple fresh results-tag name")
    try:
        configurations = cases(args.matrix, args.block_dim, args.rows_per_tile, args.dec_enc_schedule, args.gym_root)
    except ValueError as error:
        parser.error(str(error))
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    report_path = args.root / ("matrix_" + args.results_tag + ".json")
    if report_path.exists():
        parser.error("Matrix report already exists; choose a new root or --results-tag")
    report = {"schema_version": 1, "phase": args.phase, "matrix": args.matrix, "status": "running", "cases": []}
    for config in configurations:
        name, cmd = command(args, config)
        print(f"\n=== DecEnc {args.phase}: {name} ===", flush=True)
        result = subprocess.run(cmd, check=False)
        report["cases"].append({"name": name, "configuration": config, "command": cmd, "returncode": result.returncode})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed" if all(case["returncode"] == 0 for case in report["cases"]) else "failed"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "report": str(report_path)}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
