"""Compiler-host entry point: build, transfer, execute on 5102, retrieve and compare."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FRAMEWORK = Path(os.environ.get("PYPTO_SOURCE_ROOT", REPO.parent / "pypto")).resolve()
DEPENDENCIES = {name: name for name in ("numpy", "decorator", "scipy", "psutil", "sympy", "onnx")}
DEPENDENCIES["attr"] = "attrs"  # The distribution is attrs; CANN imports the attr module.


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _relative_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Bundle paths must be relative: {name!r}")
    return root / relative


def _build_interface(build_dir, output):
    if any(
        name == "pypto" or name.startswith("pypto.") or name == "pypto_pro" or name.startswith("pypto_pro.")
        for name in sys.modules
    ):
        raise RuntimeError("Rebuild must happen before importing PyPTO; start a fresh run_profile process")
    build_dir = Path(build_dir).expanduser().resolve(strict=True)
    cache = (build_dir / "CMakeCache.txt").read_text()
    source = next(
        (line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")),
        None,
    )
    if source is None or Path(source).resolve() != FRAMEWORK:
        raise ValueError("--build-dir must be the selected PyPTO checkout's configured CMake build directory")
    library = build_dir / "output/lib/libtile_fwk_interface.so"
    installed = FRAMEWORK / "python/pypto/lib/libtile_fwk_interface.so"
    if not installed.parent.is_dir():
        raise ValueError("Install this checkout in editable mode before using --build-dir")
    log = output / "interface_build.log"
    with log.open("w") as stream:
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "tile_fwk_interface", "-j4"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode or not library.is_file():
        raise RuntimeError(f"Interface rebuild failed; see {log}")
    shutil.copy2(library, installed)
    print(f"Updated editable interface: {installed}", flush=True)


def _make_bundle(case_path, om_path, binding_path, destination):
    from .host import create_execution

    destination.mkdir()
    reserved = {"execution.json", "model.om", "runner.py", "README.txt", "bundle.json"}
    for source, name in ((om_path, "model.om"), (HERE / "runner.py", "runner.py")):
        shutil.copy2(source, destination / name)
    case = json.loads(case_path.read_text())
    request = create_execution(case_path, binding_path)
    (destination / "execution.json").write_text(json.dumps(request, indent=2) + "\n")
    seen = set()
    for record in case["inputs"]:
        name = record["file"]
        if name in seen or name in reserved:
            raise ValueError(f"Repeated or reserved input file: {name}")
        seen.add(name)
        source, target = _relative_file(case_path.parent, name), _relative_file(destination, name)
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"Input changed while bundling: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "README.txt").write_text(
        "5102 native Pro execution-only bundle\n"
        "Python 3.7.9 or later plus the board Ascend runtime/driver.\n"
        "Input generation, SSH/SCP and numerical acceptance run on the compiler host.\n"
        "No golden data, PyPTO, PTO sources, ONNX, toolkit or OPP is sent to the board.\n\n"
        "python3 runner.py --execution execution.json --device 0 --output ../results\n\n"
        "Exit 0/status=executed means execution completed. It is not an accuracy verdict.\n"
    )
    files = {
        str(path.relative_to(destination)): _sha256(path) for path in sorted(destination.rglob("*")) if path.is_file()
    }
    (destination / "bundle.json").write_text(json.dumps({"schema_version": 2, "files": files}, indent=2) + "\n")


def build(args):
    output = _fresh(args.output)
    if args.build_dir:
        _build_interface(args.build_dir, output)
    missing = [package for module, package in DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "Missing compiler Python dependencies: "
            + ", ".join(missing)
            + ". In the active compiler environment, run: python3 -m pip install "
            + " ".join(DEPENDENCIES.values())
        )
    # These imports must remain after the optional C++ rebuild, and out of the board-side run path.
    from pypto_pro.runtime.device_compile import compile_device_kernel
    from pypto_pro_atc.build import build_adapter
    from pypto_pro_atc.network import build_network_adapter

    from .provider import load_provider

    provider_path, provider = load_provider(args.provider, args.gym_root)
    options = dict(vars(args))
    if args.case_config:
        extra = json.loads(args.case_config.read_text())
        if not isinstance(extra, dict):
            raise ValueError("--case-config must contain a JSON object")
        options.update(extra)
    spec = provider.build_spec(options)
    kernels = spec["kernels"]
    if not isinstance(kernels, dict) or not kernels:
        raise ValueError("Provider must supply at least one Pro kernel")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in kernels):
        raise ValueError("Provider kernel names must be simple identifiers")
    natives = {}
    graph = spec.get("graph_spec")
    if graph is not None and (not isinstance(graph, dict) or not graph):
        raise ValueError("Provider graph_spec must be a nonempty graph or None")
    if graph is None and len(kernels) != 1:
        raise ValueError("Multiple kernels require an explicit provider graph_spec")
    if any(not callable(spec.get(name)) for name in ("make_case", "bind_case")):
        raise ValueError("Provider must supply make_case and bind_case callbacks")
    for name, kernel in kernels.items():
        print(f"Compiling native Pro {name} for dav-510r2", flush=True)
        native = compile_device_kernel(
            kernel,
            output / "native" / name if graph else output / "native",
            toolkit_root=args.toolkit_root,
            pto_root=args.pto_root,
            block_dim=spec.get("block_dim", 1),
        )
        if spec.get("annotate_manifest"):
            manifest = json.loads(native.read_text())
            spec["annotate_manifest"](manifest)
            native.write_text(json.dumps(manifest, indent=2) + "\n")
        natives[name] = native
    if graph:
        adapter_path = build_network_adapter(
            natives,
            graph,
            toolkit_root=args.toolkit_root,
            target_soc=args.target_soc,
            output=output / "atc",
            run_atc=True,
        )
        embedding = "all_frozen_device_elfs_found"
    else:
        native = next(iter(natives.values()))
        adapter_path = build_adapter(
            native,
            toolkit_root=args.toolkit_root,
            target_soc=args.target_soc,
            output=output / "atc",
            run_atc=True,
        )
        embedding = "exact_frozen_device_elf_found"
    adapter = json.loads(adapter_path.read_text())
    if (
        adapter.get("status") != "om-compiled"
        or adapter.get("validation", {}).get("kernel_embedding") != embedding
        or adapter.get("validation", {}).get("host_portability") != "passed"
    ):
        raise RuntimeError("ATC did not prove a device-only OM containing the frozen native device ELF")
    case_path = Path(spec["make_case"](adapter_path, output / "case"))
    if case_path.resolve() != output / "case/case.json":
        raise ValueError("Provider must return the generated output/case/case.json")
    om_path, binding_path = Path(adapter["om"]["path"]), output / "case/binding.json"
    spec["bind_case"](case_path, om_path, binding_path, adapter_path)
    bundle = output / "bundle"
    _make_bundle(case_path, om_path, binding_path, bundle)
    from .host import validate_build

    validate_build(output)
    if spec.get("analyze"):
        metrics = spec["analyze"](natives, adapter_path)
        (output / "codegen_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if _sha256(provider_path) != provider.__provider_sha256__:
        raise ValueError("Provider changed during compilation; rebuild from a stable checkout")
    report = {
        "schema_version": 1,
        "status": "bundle-built",
        "operator": options.get("op", args.op),
        "provider": str(provider_path),
        "provider_sha256": provider.__provider_sha256__,
        "target_soc": args.target_soc,
        "bundle": str(bundle),
        "device_execution": "not_run",
        "runtime_dependencies": "device_kernel_only",
        "adapter_manifest": str(adapter_path),
        "case": str(case_path),
    }
    if spec.get("analyze"):
        report["codegen_metrics"] = str(output / "codegen_metrics.json")
    (output / "build_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def run(args):
    from . import host

    output = _fresh(args.output)
    try:
        if args.command == "run":
            report = host.remote_run(args, output)
        elif args.command == "profile":
            from .profiling import remote_profile

            report = remote_profile(args, output)
        else:
            report = host.evaluate(args.build, args.results)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        report = {"schema_version": 1, "status": "failed", "precision_checked_on": "compiler_host", "error": str(error)}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    display = report
    if args.command == "profile":
        performance = report.get("performance") or {}
        display = {
            "status": report["status"],
            "operator": report.get("operator"),
            "result": str(output / "result.json"),
            "node_duration_us": {name: node["statistics_us"] for name, node in performance.get("nodes", {}).items()},
            "device_task_sum_us": performance.get("device_task_sum_us", {}).get("statistics"),
            "statistics_csv": performance.get("statistics_csv", {}).get("path"),
            "errors": report.get("errors", report.get("error")),
        }
    print(json.dumps(display, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compiler = commands.add_parser("build", help="Compile one Pro operator into a fresh runtime bundle")
    compiler.add_argument("--op", required=True, help="Operator name accepted by the gym provider")
    compiler.add_argument("--provider", type=Path, help="Gym test provider.py; defaults to the native_5102 suite")
    compiler.add_argument("--gym-root", type=Path, help="Source checkout containing the operator and golden")
    compiler.add_argument("--case-config", type=Path, help="JSON options passed to an existing custom provider")
    compiler.add_argument(
        "--dec-enc-schedule",
        choices=("baseline", "optimized", "pipelined"),
        default="pipelined",
        help="DecEnc schedule; baseline preserves 8dfc18d72 for same-toolchain A/B",
    )
    compiler.add_argument("--height", type=int, default=540, help="DecEnc H: 540 times 1..4")
    compiler.add_argument("--width", type=int, default=960, help="DecEnc W: 960 times 1..4")
    compiler.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32", help="DecEnc I/O dtype")
    compiler.add_argument("--block-dim", type=int, default=8, help="DecEnc coupled AiCore blocks (1..8)")
    compiler.add_argument(
        "--rows-per-tile", type=int, help="DecEnc rows; pipeline defaults fp32=6/fp16=12, older schedules=3"
    )
    compiler.add_argument("--toolkit-root", type=Path, required=True)
    compiler.add_argument("--pto-root", type=Path, required=True)
    compiler.add_argument(
        "--target-soc", default="MC62CM12AA", help="Explicit ATC SoC; default is the verified MC62 target"
    )
    compiler.add_argument("--output", type=Path, required=True, help="New build directory; it must not exist")
    compiler.add_argument("--build-dir", type=Path, help="Optional configured CMake directory for this checkout")
    execution = commands.add_parser(
        "run", help="Transfer, execute remotely, retrieve and compare on this compiler host"
    )
    profiler = commands.add_parser(
        "profile", help="Upload, collect msprof, retrieve, check accuracy and parse on this host"
    )
    for command in (execution, profiler):
        command.add_argument("--build", type=Path, required=True)
        command.add_argument("--host", required=True, help="SSH host or config alias reachable from the compiler host")
        command.add_argument("--remote-dir", required=True, help="New absolute Linux task directory")
        command.add_argument("--runtime-env", help="Optional Ascend runtime environment script on the board")
        command.add_argument("--python", default="python3", help="Board Python executable, supporting 3.7.9+")
        command.add_argument("--device", type=int, default=0)
        command.add_argument("--timeout", type=int, default=600, help="Timeout per SSH/SCP operation in seconds")
        command.add_argument("--output", type=Path, required=True, help="New compiler-side results directory")
    profiler.add_argument("--warmup", type=int, default=5)
    profiler.add_argument("--repeat", type=int, default=50)
    profiler.add_argument("--msprof", default="/var/msprof", help="Absolute board collector path; Fused command style")
    profiler.add_argument("--msprof-py", type=Path, required=True, help="Compiler-host analysis/msprof/msprof.py path")
    comparison = commands.add_parser("compare", help="Compare already retrieved raw results on the compiler host")
    comparison.add_argument("--build", type=Path, required=True)
    comparison.add_argument(
        "--results", type=Path, required=True, help="Retrieved board execution.json/raw outputs directory"
    )
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return build(args) if args.command == "build" else run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
