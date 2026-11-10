"""Compiler-side transfer, artifact validation and numerical acceptance for the 1952 runner."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess

IO_CHUNK_BYTES = 1024 * 1024
DTYPES = {"fp16": (2, "<e", 1), "fp32": (4, "<f", 0)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(record):
    if record.get("dtype") not in DTYPES:
        raise ValueError("Supported tensor dtypes are fp16 and fp32")
    shape = record.get("shape")
    if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("Tensor shape must contain positive integers")
    return math.prod(shape) * DTYPES[record["dtype"]][0]


def relative_file(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Expected a relative artifact path: " + str(name))
    result = (Path(root) / path).resolve(strict=True)
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes its directory")
    return result


def create_execution(case_file, binding_file):
    case = json.loads(Path(case_file).read_text())
    binding = json.loads(Path(binding_file).read_text())
    if binding.get("case_sha256") != sha256(case_file) or binding.get("runtime_dependencies") != "device_kernel_only":
        raise ValueError("Execution request requires a verified case/model binding")
    return {
        "schema_version": 1,
        "kind": "pypto_pro_execution_request",
        "runtime_dependencies": "device_kernel_only",
        "om": {"file": "model.om", "sha256": binding["om_sha256"]},
        "inputs": [{"name": item["name"], "file": item["file"], "sha256": item["sha256"]} for item in case["inputs"]],
        "output": {
            "name": case["outputs"][0]["name"],
            "model_names": binding["output_name_bindings"][0]["om_declared_names"],
        },
    }


def validate_build(build_dir):
    root = Path(build_dir).resolve(strict=True)
    case_path, binding_path, bundle = root / "case/case.json", root / "case/binding.json", root / "bundle"
    case, binding = json.loads(case_path.read_text()), json.loads(binding_path.read_text())
    if (
        binding.get("schema_version") != 2
        or binding.get("case_sha256") != sha256(case_path)
        or binding.get("om_sha256") != sha256(bundle / "model.om")
    ):
        raise ValueError("Compiler-side case/model binding changed")
    from pypto_pro_atc.audit import audit_om_binary

    if audit_om_binary(bundle / "model.om") != binding.get("host_portability"):
        raise ValueError("OM portability evidence changed")
    inventory = json.loads((bundle / "bundle.json").read_text())
    if (
        inventory.get("schema_version") != 2
        or not {"runner.py", "execution.json", "model.om"} <= inventory.get("files", {}).keys()
    ):
        raise ValueError("Rebuild the execution-only bundle with the current scripts")
    for name, expected in inventory["files"].items():
        if sha256(relative_file(bundle, name)) != expected:
            raise ValueError("Runtime bundle SHA256 mismatch: " + name)
    request = create_execution(case_path, binding_path)
    if request != json.loads((bundle / "execution.json").read_text()):
        raise ValueError("Execution request differs from compiler-side case/binding")
    allowed = {"runner.py", "execution.json", "model.om", "README.txt"} | {item["file"] for item in case["inputs"]}
    if set(inventory["files"]) != allowed:
        raise ValueError("Runtime bundle must contain only the executor, OM and inputs")
    actual_files = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
    if actual_files != allowed | {"bundle.json"} or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("Runtime bundle contains unlisted files or symlinks; refusing to upload")
    for item in case["inputs"] + case["outputs"]:
        payload = relative_file(case_path.parent, item["file"])
        if (
            sha256(payload) != item["sha256"]
            or payload.stat().st_size != tensor_bytes(item)
            or item.get("bytes") != payload.stat().st_size
        ):
            raise ValueError("Compiler-side tensor changed: " + item["name"])
    for tolerance in ("atol", "rtol"):
        value = float(case[tolerance])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid compiler-side tolerance")
    return root, case, binding, request


def _compare_chunks(chunks, size, atol, rtol, dtype):
    if dtype not in DTYPES:
        raise ValueError("Supported tensor dtypes are fp16 and fp32")
    width, format_, _ = DTYPES[dtype]
    if size <= 0 or size % width:
        raise ValueError("Output byte size mismatch")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Invalid comparison tolerance")
    try:
        import numpy as np
    except ImportError:
        np = None
    mismatches, nonfinite, maximum = 0, 0, 0.0
    consumed = 0
    for actual, golden in chunks:
        if len(actual) != len(golden) or len(actual) % width:
            raise ValueError("Output byte size mismatch")
        consumed += len(actual)
        if not actual:
            continue
        if np is not None:
            # FP64 arithmetic matches Python float, including FP32 subtraction near its extrema.
            got = np.frombuffer(actual, dtype="<f" + str(width)).astype(np.float64)
            want = np.frombuffer(golden, dtype="<f" + str(width)).astype(np.float64)
            finite = np.isfinite(got) & np.isfinite(want)
            invalid = int(np.count_nonzero(~finite))
            with np.errstate(invalid="ignore", over="ignore"):
                differences = np.abs(got - want)
                mismatches += int(np.count_nonzero(~finite | (differences > atol + rtol * np.abs(want))))
            nonfinite += invalid
            maximum = max(maximum, math.inf if invalid else float(differences.max()))
        else:
            for (got,), (want,) in zip(struct.iter_unpack(format_, actual), struct.iter_unpack(format_, golden)):
                finite = math.isfinite(got) and math.isfinite(want)
                nonfinite += not finite
                difference = abs(got - want) if finite else math.inf
                mismatches += not finite or difference > atol + rtol * abs(want)
                maximum = max(maximum, difference)
    if consumed != size:
        raise ValueError("Output changed during comparison")
    return {
        "passed": mismatches == 0,
        "elements": size // width,
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "max_absolute_error": maximum if math.isfinite(maximum) else None,
    }


def compare(actual, golden, atol=0, rtol=0, dtype="fp16"):
    """Compare in-memory samples; deployed tensors use compare_files instead."""
    if len(actual) != len(golden):
        raise ValueError("Output byte size mismatch")
    return _compare_chunks(((actual, golden),), len(actual), atol, rtol, dtype)


def compare_files(actual, golden, atol=0, rtol=0, dtype="fp16"):
    """Compare tensors with bounded memory, including FP32 images larger than 1 GB."""
    if dtype not in DTYPES:
        raise ValueError("Supported tensor dtypes are fp16 and fp32")
    actual, golden = Path(actual), Path(golden)
    size = actual.stat().st_size
    if size != golden.stat().st_size:
        raise ValueError("Output byte size mismatch")
    width = DTYPES[dtype][0]
    chunk_bytes = max(width, IO_CHUNK_BYTES // width * width)
    with actual.open("rb") as actual_stream, golden.open("rb") as golden_stream:

        def chunks():
            while True:
                left, right = actual_stream.read(chunk_bytes), golden_stream.read(chunk_bytes)
                if not left and not right:
                    break
                yield left, right

        return _compare_chunks(chunks(), size, atol, rtol, dtype)


def files_equal(first, second):
    if Path(first).stat().st_size != Path(second).stat().st_size:
        return False
    with Path(first).open("rb") as left, Path(second).open("rb") as right:
        while True:
            a, b = left.read(IO_CHUNK_BYTES), right.read(IO_CHUNK_BYTES)
            if a != b:
                return False
            if not a:
                return True


def evaluate(build_dir, results_dir):
    root, case, binding, request = validate_build(build_dir)
    results = Path(results_dir).resolve(strict=True)
    report = json.loads((results / "execution.json").read_text())
    if (
        report.get("kind") != "pypto_pro_execution_result"
        or report.get("status") != "executed"
        or report.get("schema_version") != 1
        or report.get("execution_calls") != 2
    ):
        raise ValueError("Board did not complete both executions: " + str(report.get("error", report.get("status"))))
    if (
        report.get("request_sha256") != sha256(root / "bundle/execution.json")
        or report.get("om_sha256") != binding["om_sha256"]
    ):
        raise ValueError("Board executed a different request/OM")
    inputs = report.get("inputs", [])
    if len(inputs) != len(case["inputs"]) or len({item["name"] for item in inputs}) != len(inputs):
        raise ValueError("Board input descriptors are incomplete")
    actual_inputs = {item["name"]: item for item in inputs}
    for expected in case["inputs"]:
        actual = actual_inputs.get(expected["name"], {})
        if (
            any(actual.get(key) != expected[key] for key in ("sha256", "shape", "bytes"))
            or actual.get("acl_dtype") != DTYPES[expected["dtype"]][2]
        ):
            raise ValueError("Board input contract differs: " + expected["name"])
    output = report.get("output", {})
    expected = case["outputs"][0]
    if (
        output.get("shape") != expected["shape"]
        or output.get("acl_dtype") != DTYPES[expected["dtype"]][2]
        or output.get("bytes") != expected["bytes"]
        or output.get("name") != expected["name"]
        or output.get("model_name") not in request["output"]["model_names"]
    ):
        raise ValueError("Board output descriptor differs from the compiled case")
    rounds = report.get("rounds", [])
    if len(rounds) != 2:
        raise ValueError("Board results require two raw output files")
    payloads = []
    for index, item in enumerate(rounds):
        name = ("actual.bin", "actual_second.bin")[index]
        if item.get("index") != index + 1 or item.get("file") != name or item.get("sentinel") != (127, 128)[index]:
            raise ValueError("Board output rounds do not match the execution protocol")
        path = relative_file(results, name)
        if (
            sha256(path) != item.get("sha256")
            or path.stat().st_size != expected["bytes"]
            or item.get("bytes") != expected["bytes"]
        ):
            raise ValueError("Copied board output changed: " + name)
        payloads.append(path)
    golden = relative_file(root / "case", expected["file"])
    comparisons = [compare_files(raw, golden, case["atol"], case["rtol"], expected["dtype"]) for raw in payloads]
    same = files_equal(*payloads)
    passed = same and all(item["passed"] for item in comparisons)
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "operator": case["operator"],
        "device_execution": "executed",
        "precision_checked_on": "compiler_host",
        "om_sha256": binding["om_sha256"],
        "request_sha256": report["request_sha256"],
        "dual_sentinel_consistent": same,
        "comparisons": comparisons,
        "atol": case["atol"],
        "rtol": case["rtol"],
    }


def remote_target(args):
    if args.device < 0 or args.timeout <= 0:
        raise ValueError("Device ID must be nonnegative and SSH/SCP timeout must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_@.:[\]-]+", args.host) or args.host.startswith("-"):
        raise ValueError("Use a hostname, user@host or SSH config alias")
    remote = PurePosixPath(args.remote_dir)
    if (
        not remote.is_absolute()
        or ".." in remote.parts
        or len(remote.parts) < 3
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_dir)
    ):
        raise ValueError("Use a new absolute Linux task directory without spaces or shell characters")
    return remote


def remote_run(args, output):
    root, _, _, _ = validate_build(args.build)
    output = Path(output)
    remote = remote_target(args)
    commands = []

    def invoke(argv, log):
        commands.append(argv)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(argv, capture_output=True, text=True, timeout=args.timeout, check=False)
        (output / log).write_text(result.stdout + result.stderr)
        return result.returncode

    setup = "mkdir -p -- " + shlex.quote(str(remote.parent)) + " && mkdir -- " + shlex.quote(str(remote))
    if invoke(["ssh", args.host, setup], "ssh_prepare.log"):
        raise RuntimeError("Remote task directory could not be created; see ssh_prepare.log")
    if invoke(["scp", "-r", str(root / "bundle"), args.host + ":" + str(remote / "bundle")], "scp_upload.log"):
        raise RuntimeError("Runtime bundle transfer failed; see scp_upload.log")
    command = "set -e; "
    if args.runtime_env:
        command += "source " + shlex.quote(args.runtime_env) + "; "
    command += (
        "cd "
        + shlex.quote(str(remote / "bundle"))
        + "; "
        + shlex.quote(args.python)
        + " runner.py --execution execution.json --device "
        + str(args.device)
        + " --output ../results"
    )
    execute_status = invoke(["ssh", args.host, "bash -lc " + shlex.quote(command)], "board_execution.log")
    collect_status = invoke(
        ["scp", "-r", args.host + ":" + str(remote / "results"), str(output / "board")], "scp_download.log"
    )
    if collect_status:
        raise RuntimeError("Could not retrieve raw board results; see board_execution.log and scp_download.log")
    if execute_status:
        raise RuntimeError("Board execution failed; raw results and logs were retrieved to the compiler host")
    return evaluate(root, output / "board")
