"""Execute an OM and save raw outputs; Python 3.7+ standard library only.

Input generation, transfer and comparison belong to the compilation host.
Exit zero means execution completed, not accuracy passed.
"""

import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
from pathlib import Path
import sys
import time

IO_CHUNK_BYTES = 1024 * 1024
DTYPES = {0: ("fp32", 4), 1: ("fp16", 2)}


class ModelDimensions(ct.Structure):
    _fields_ = [("name", ct.c_char * 128), ("dimCount", ct.c_size_t), ("dims", ct.c_int64 * 128)]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed: {result}")


def load_acl():
    acl = ct.CDLL(ctypes.util.find_library("ascendcl") or "libascendcl.so")
    ptr, size, integer, u32, text = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_uint32, ct.c_char_p
    signatures = {
        "aclInit": ([text], integer),
        "aclFinalize": ([], integer),
        "aclrtSetDevice": ([integer], integer),
        "aclrtResetDevice": ([integer], integer),
        "aclrtMalloc": ([ct.POINTER(ptr), size, integer], integer),
        "aclrtFree": ([ptr], integer),
        "aclrtMemcpy": ([ptr, size, ptr, size, integer], integer),
        "aclmdlLoadFromFile": ([text, ct.POINTER(u32)], integer),
        "aclmdlUnload": ([u32], integer),
        "aclmdlCreateDesc": ([], ptr),
        "aclmdlDestroyDesc": ([ptr], integer),
        "aclmdlGetDesc": ([ptr, u32], integer),
        "aclmdlGetNumInputs": ([ptr], size),
        "aclmdlGetNumOutputs": ([ptr], size),
        "aclmdlGetInputIndexByName": ([ptr, text, ct.POINTER(size)], integer),
        "aclmdlGetOutputNameByIndex": ([ptr, size], text),
        "aclmdlGetInputDims": ([ptr, size, ct.POINTER(ModelDimensions)], integer),
        "aclmdlGetOutputDims": ([ptr, size, ct.POINTER(ModelDimensions)], integer),
        "aclmdlGetInputDataType": ([ptr, size], integer),
        "aclmdlGetOutputDataType": ([ptr, size], integer),
        "aclmdlGetInputSizeByIndex": ([ptr, size], size),
        "aclmdlGetOutputSizeByIndex": ([ptr, size], size),
        "aclmdlCreateDataset": ([], ptr),
        "aclmdlDestroyDataset": ([ptr], integer),
        "aclCreateDataBuffer": ([ptr, size], ptr),
        "aclDestroyDataBuffer": ([ptr], integer),
        "aclmdlAddDatasetBuffer": ([ptr, ptr], integer),
        "aclmdlExecute": ([u32, ptr, ptr], integer),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(acl, name, None)
        if function is None:
            raise RuntimeError(f"libascendcl lacks required capability: {name}")
        function.argtypes, function.restype = arguments, result
    return acl


def _payload(root, record):
    relative = Path(record["file"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Execution payload paths must be relative to execution.json")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Execution payload escapes its directory")
    if not path.stat().st_size or file_sha256(path) != record["sha256"]:
        raise ValueError("Execution payload is empty or has a SHA256 mismatch: " + str(relative))
    return path


def load_execution(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    request = json.loads(raw)
    if (
        not isinstance(request, dict)
        or request.get("schema_version") != 1
        or request.get("kind") != "pypto_pro_execution_request"
        or request.get("runtime_dependencies") != "device_kernel_only"
    ):
        raise ValueError("Require a device-only schema-1 execution request generated on the compilation host")
    if any(key in request for key in ("golden", "atol", "rtol", "binding", "case")):
        raise ValueError("Execution requests must not contain host comparison data")
    om_path = _payload(path.parent, request["om"])
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Execution request requires input files")
    seen = set()
    for item in inputs:
        name = item["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Input names must be nonempty and unique")
        seen.add(name)
        item["_path"] = _payload(path.parent, item)
        item["_bytes"] = item["_path"].stat().st_size
    output = request["output"]
    if not isinstance(output, dict):
        raise ValueError("Execution output must contain its logical and audited model names")
    names = output.get("model_names", [output["name"]])
    if (
        not isinstance(output.get("name"), str)
        or not output["name"]
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Execution output requires its logical and audited model names")
    output["model_names"] = names
    request["_om_path"], request["_sha256"] = om_path, sha256(raw)
    return request


def _descriptor(acl, desc, index, direction):
    dimensions = ModelDimensions()
    _checked(getattr(acl, "aclmdlGet" + direction + "Dims")(desc, index, ct.byref(dimensions)), direction + "Dims")
    size = int(getattr(acl, "aclmdlGet" + direction + "SizeByIndex")(desc, index))
    dtype = int(getattr(acl, "aclmdlGet" + direction + "DataType")(desc, index))
    shape = list(dimensions.dims[: dimensions.dimCount])
    if dimensions.dimCount == 0 or dimensions.dimCount > 128 or size <= 0 or any(dim <= 0 for dim in shape):
        raise ValueError("Invalid runtime tensor dimensions or byte size")
    if dtype not in DTYPES:
        raise ValueError("Runtime tensor dtype must be FP16 or FP32")
    elements = 1
    for dim in shape:
        elements *= dim
    if size != elements * DTYPES[dtype][1]:
        raise ValueError("Runtime tensor shape/dtype/byte size are inconsistent")
    return {
        "model_index": index,
        "shape": shape,
        "acl_dtype": dtype,
        "dtype": DTYPES[dtype][0],
        "bytes": size,
    }


def _copy_input(acl, address, tensor):
    size = tensor["_bytes"]
    host = ct.create_string_buffer(min(size, IO_CHUNK_BYTES))
    view = memoryview(host).cast("B")
    digest, copied = hashlib.sha256(), 0
    with tensor["_path"].open("rb") as stream:
        while copied < size:
            count = stream.readinto(view[: min(len(host), size - copied)])
            if not count:
                raise ValueError("Input changed during H2D: " + tensor["name"])
            digest.update(view[:count])
            _checked(
                acl.aclrtMemcpy(ct.c_void_p(address.value + copied), count, ct.cast(host, ct.c_void_p), count, 1),
                "input H2D",
            )
            copied += count
        if stream.read(1) or digest.hexdigest() != tensor["sha256"]:
            raise ValueError("Input changed during H2D: " + tensor["name"])


def _prefill_output(acl, address, size, sentinel):
    count = min(size, IO_CHUNK_BYTES)
    host = ct.create_string_buffer(bytes([sentinel]) * count, count)
    for offset in range(0, size, count):
        length = min(count, size - offset)
        _checked(
            acl.aclrtMemcpy(ct.c_void_p(address.value + offset), length, ct.cast(host, ct.c_void_p), length, 1),
            "output prefill",
        )


def _save_output(acl, address, size, path):
    """Copy through one bounded host buffer; hash the same bytes written to disk."""
    host = ct.create_string_buffer(min(size, IO_CHUNK_BYTES))
    view, digest = memoryview(host).cast("B"), hashlib.sha256()
    with Path(path).open("wb") as stream:
        for offset in range(0, size, len(host)):
            count = min(len(host), size - offset)
            _checked(
                acl.aclrtMemcpy(ct.cast(host, ct.c_void_p), count, ct.c_void_p(address.value + offset), count, 2),
                "output D2H",
            )
            chunk = view[:count]
            stream.write(chunk)
            digest.update(chunk)
    return {"file": Path(path).name, "sha256": digest.hexdigest(), "bytes": size}


def _profiling_execution(warmup, repeat):
    if type(warmup) is not int or type(repeat) is not int or warmup < 0 or repeat < 0:
        raise ValueError("Warmup and repeat must be nonnegative integers")
    if warmup and not repeat:
        raise ValueError("Profiling warmup requires repeat > 0")
    return {
        "enabled": repeat > 0,
        "status": "pending" if repeat else "not_requested",
        "warmup": warmup,
        "repeat": repeat,
        "completed_warmup": 0,
        "completed_repeat": 0,
        "correctness_calls": 0,
        "total_execution_calls": 0,
        "attempted_execution_calls": 0,
        "discard_first_executions": 2 + warmup,
        "host_wall_seconds": [],
        "timing_scope": "synchronous_aclmdlExecute_host_wall",
        "count_scope": "model_execution_not_individual_kernel",
    }


def execute(acl, request, output_dir, device=0, report=None, warmup=0, repeat=0):
    """Save two raw rounds, then optionally warm up and time synchronous model calls."""
    report = {} if report is None else report
    report.update(inputs=[], rounds=[], execution_calls=0)
    profile = _profiling_execution(warmup, repeat)
    report["profiling_execution"] = profile
    initialized = selected = False
    model_id, desc = None, None
    datasets, buffers, allocations = [], [], []
    try:
        _checked(acl.aclInit(None), "aclInit")
        initialized = True
        _checked(acl.aclrtSetDevice(device), "aclrtSetDevice")
        selected = True
        model = ct.c_uint32()
        _checked(acl.aclmdlLoadFromFile(str(request["_om_path"]).encode(), ct.byref(model)), "aclmdlLoadFromFile")
        model_id = model.value
        desc = acl.aclmdlCreateDesc()
        if not desc:
            raise RuntimeError("aclmdlCreateDesc returned null")
        _checked(acl.aclmdlGetDesc(desc, model_id), "aclmdlGetDesc")
        count = int(acl.aclmdlGetNumInputs(desc))
        if count != len(request["inputs"]) or acl.aclmdlGetNumOutputs(desc) != 1:
            raise ValueError("Model input/output count differs from the execution request")
        ordered = [None] * count
        for tensor in request["inputs"]:
            index = ct.c_size_t()
            _checked(acl.aclmdlGetInputIndexByName(desc, tensor["name"].encode(), ct.byref(index)), "input name lookup")
            if index.value >= count or ordered[index.value] is not None:
                raise ValueError("Model input names are not a bijection")
            metadata = _descriptor(acl, desc, index.value, "Input")
            if metadata["bytes"] != tensor["_bytes"]:
                raise ValueError("Input bytes differ from runtime model: " + tensor["name"])
            ordered[index.value] = tensor
            report["inputs"].append(dict(metadata, name=tensor["name"], sha256=tensor["sha256"]))
        report["inputs"].sort(key=lambda item: item["model_index"])
        actual_name = acl.aclmdlGetOutputNameByIndex(desc, 0)
        if actual_name not in [name.encode() for name in request["output"]["model_names"]]:
            raise ValueError("Runtime output name is absent from the audited execution request")
        metadata = _descriptor(acl, desc, 0, "Output")
        report["output"] = dict(metadata, name=request["output"]["name"], model_name=actual_name.decode())
        size = metadata["bytes"]
        for tensor in ordered + [None]:
            allocation_size = tensor["_bytes"] if tensor is not None else size
            address = ct.c_void_p()
            _checked(acl.aclrtMalloc(ct.byref(address), allocation_size, 0), "aclrtMalloc")
            allocations.append(address)
            if tensor is not None:
                _copy_input(acl, address, tensor)
            buffer = acl.aclCreateDataBuffer(address, allocation_size)
            if not buffer:
                raise RuntimeError("aclCreateDataBuffer returned null")
            buffers.append(buffer)
        for _ in range(2):
            dataset = acl.aclmdlCreateDataset()
            if not dataset:
                raise RuntimeError("aclmdlCreateDataset returned null")
            datasets.append(dataset)
        for buffer in buffers[:-1]:
            _checked(acl.aclmdlAddDatasetBuffer(datasets[0], buffer), "input dataset")
        _checked(acl.aclmdlAddDatasetBuffer(datasets[1], buffers[-1]), "output dataset")
        for number, sentinel in enumerate((0x7F, 0x80), 1):
            _prefill_output(acl, allocations[-1], size, sentinel)
            profile["attempted_execution_calls"] += 1
            started = time.perf_counter()
            _checked(acl.aclmdlExecute(model_id, datasets[0], datasets[1]), "aclmdlExecute")
            elapsed = time.perf_counter() - started
            report["execution_calls"] += 1
            profile["correctness_calls"] += 1
            profile["total_execution_calls"] += 1
            filename = "actual.bin" if number == 1 else "actual_second.bin"
            saved = _save_output(acl, allocations[-1], size, Path(output_dir) / filename)
            report["rounds"].append(dict(saved, index=number, sentinel=sentinel, elapsed_seconds=elapsed))
        if profile["enabled"]:
            profile["status"] = "warmup"
            for _ in range(warmup):
                profile["attempted_execution_calls"] += 1
                _checked(acl.aclmdlExecute(model_id, datasets[0], datasets[1]), "aclmdlExecute warmup")
                profile["completed_warmup"] += 1
                profile["total_execution_calls"] += 1
            profile["status"] = "measuring"
            for _ in range(repeat):
                profile["attempted_execution_calls"] += 1
                started = time.perf_counter()
                result = acl.aclmdlExecute(model_id, datasets[0], datasets[1])
                elapsed = time.perf_counter() - started
                _checked(result, "aclmdlExecute profile")
                profile["completed_repeat"] += 1
                profile["total_execution_calls"] += 1
                profile["host_wall_seconds"].append(elapsed)
            profile["final_output"] = _save_output(acl, allocations[-1], size, Path(output_dir) / "profile_actual.bin")
            profile["status"] = "completed"
        return report
    finally:
        if profile["enabled"] and profile["status"] != "completed":
            profile["status"] = "failed"
        for dataset in datasets:
            acl.aclmdlDestroyDataset(dataset)
        for buffer in buffers:
            acl.aclDestroyDataBuffer(buffer)
        for address in allocations:
            acl.aclrtFree(address)
        if desc:
            acl.aclmdlDestroyDesc(desc)
        if model_id is not None:
            acl.aclmdlUnload(model_id)
        if selected:
            acl.aclrtResetDevice(device)
        if initialized:
            acl.aclFinalize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True, help="Input-only execution request JSON")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for raw outputs and execution evidence"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--warmup", type=int, default=0, help="Extra untimed model calls after the two raw-output rounds"
    )
    parser.add_argument(
        "--repeat", type=int, default=0, help="Timed model calls; zero preserves execution-only behavior"
    )
    args = parser.parse_args(argv)
    report = {
        "schema_version": 1,
        "kind": "pypto_pro_execution_result",
        "status": "failed",
        "device": args.device,
        "python_version": sys.version.split()[0],
        "inputs": [],
        "rounds": [],
        "execution_calls": 0,
    }
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print("Cannot create a new output directory: " + str(exc), file=sys.stderr)
        return 1
    try:
        if args.device < 0:
            raise ValueError("Device ID must be nonnegative")
        report["profiling_execution"] = _profiling_execution(args.warmup, args.repeat)
        request = load_execution(args.execution)
        report.update(request_sha256=request["_sha256"], om_sha256=request["om"]["sha256"])
        execute(load_acl(), request, args.output, args.device, report, warmup=args.warmup, repeat=args.repeat)
        report["status"] = "executed"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report["error"] = str(exc)
    (args.output / "execution.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
