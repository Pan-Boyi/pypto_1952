"""FP16 native network provider for the existing pypto_1952 build/run/profile CLI."""
import hashlib
import json
from pathlib import Path

from pypto_1952.isp_1952_validation import GEMM_KERNEL
from pypto_1952.provider import load_provider, resolve_gym_root

GRAPH_SPEC = {
    "name": "GEMMFusedKernel",
    "inputs": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "w1t", "b1", "w2t", "b2", "constants"],
    "outputs": ["y"],
    "nodes": [
        {"name": "fc1", "kernel": "first_gemm", "inputs": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "w1t"], "output": "raw"},
        {"name": "activation", "kernel": "bias_leaky", "inputs": ["raw", "b1"], "output": "hidden"},
        {"name": "fc2", "kernel": "second_gemm", "inputs": ["hidden", "w2t"], "output": "delta"},
        {"name": "params", "kernel": "postprocess", "inputs": ["delta", "b2", "constants"], "output": "y"},
    ],
}


def load_kernel(gym_root=None):
    _, module = load_provider(resolve_gym_root(gym_root) / GEMM_KERNEL)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(adapter):
    result = {}
    if set(adapter["kernels"]) != {"first_gemm", "bias_leaky", "second_gemm", "postprocess"}:
        raise ValueError("Expected the four-stage grouped FP16 network")
    for name, item in adapter["kernels"].items():
        manifest, obj = Path(item["frozen_manifest"]), Path(item["frozen_object"])
        if sha256(manifest) != item["native_manifest_sha256"] or sha256(obj) != item["native_object_sha256"]:
            raise ValueError("Frozen kernel changed: " + name)
        description = json.loads(manifest.read_text())
        if any(p["dtype"] != "fp16" for p in description["parameters"]):
            raise ValueError("Every model input, output and intermediate tensor must be FP16")
        result[name] = {"manifest_sha256": sha256(manifest), "object_sha256": sha256(obj)}
    return result


def make_case(adapter_path, output):
    from pypto_1952.isp_1952_validation.gemm_fused_kernel import packing, reference as ref

    adapter = json.loads(Path(adapter_path).read_text())
    native = identities(adapter)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty case directory")
    inputs, weights = ref.make_example()
    packed = packing.pack_weights(weights)
    golden = packing.pack_outputs(ref.reference(inputs, weights))
    records = []
    for name, value in zip(packing.INPUT_NAMES + ("y",), inputs + list(packed) + [golden]):
        file = output / (name + ".bin")
        value.detach().cpu().numpy().astype("<f2").tofile(file)
        records.append({"name": name, "shape": list(value.shape), "dtype": "fp16", "file": file.name,
                        "bytes": file.stat().st_size, "sha256": sha256(file)})
    config = json.loads(Path(adapter["kernels"]["first_gemm"]["frozen_manifest"]).read_text())["gemm_fused_kernel_configuration"]
    case = {"schema_version": 1, "operator": "GEMMFusedKernel", "inputs": records[:-1], "outputs": records[-1:],
            "atol": 0.003, "rtol": 0.005, "native_kernels": native, "configuration": config,
            "golden": "Independent PyTorch FP16 reference; FP16 output after each GEMM, bias and nonlinear operation",
            "seed": 20261111, "negative_slope": 0.01, "weights": "eight independent sets",
            "accumulator": "Native 5102 FP16 operands / INT32 fixed-point MAD / FP16 DEQF16 writeback",
            "output_layout": {"shape": [68, 16], "row_starts": list(ref.ROW_STARTS),
                              "batches": list(ref.BATCHES), "widths": list(ref.WIDTHS), "padding": 0}}
    path = output / "case.json"
    path.write_text(json.dumps(case, indent=2) + "\n")
    return path


def bind_case(case_file, om_file, output, adapter_path):
    from pypto_pro_atc.audit import audit_om_binary
    from pypto_pro_atc.network import validate_network_launch_audit, validate_network_spec

    case_file, om_file, adapter_path = map(Path, (case_file, om_file, adapter_path))
    case, adapter = json.loads(case_file.read_text()), json.loads(adapter_path.read_text())
    if (adapter.get("kind") != "native-network" or adapter.get("status") != "om-compiled"
            or adapter.get("om", {}).get("sha256") != sha256(om_file)
            or case["native_kernels"] != identities(adapter)
            or adapter["graph_spec"] != GRAPH_SPEC
            or any(adapter.get("validation", {}).get(k) != v for k, v in
                   (("atc", "passed"), ("kernel_embedding", "all_frozen_device_elfs_found"),
                    ("host_portability", "passed"), ("launch_abi", "passed")))):
        raise ValueError("Case, native FP16 network and OM identities differ")
    audit_path = adapter_path.parent / "om_audit.json"
    audit = json.loads(audit_path.read_text())
    portability = audit_om_binary(om_file)
    if (adapter["audit"]["sha256"] != sha256(audit_path) or audit.get("host_portability") != portability
            or audit.get("om_sha256") != sha256(om_file)):
        raise ValueError("Network OM audit changed")
    manifests = {name: json.loads(Path(item["frozen_manifest"]).read_text())
                 for name, item in adapter["kernels"].items()}
    validate_network_launch_audit(audit, manifests, adapter["graph_spec"], adapter["kernels"], adapter["target_soc"])
    tensors = validate_network_spec(manifests, adapter["graph_spec"])
    for record in case["inputs"] + case["outputs"]:
        declared = tensors[record["name"]]
        if declared["shape"] != record["shape"] or declared["dtype"] != "fp16":
            raise ValueError("Case tensor differs from frozen network: " + record["name"])
    for item in adapter["kernels"].values():
        if om_file.read_bytes().find(Path(item["frozen_object"]).read_bytes()) < 0:
            raise ValueError("OM lacks a complete frozen native ELF")
    declared = audit["graph"]["outputs"][0]
    names = list(dict.fromkeys(v for v in (declared.get("logical_name"), declared.get("producer_ref"),
                                           declared.get("model_declared_name"), "y") if v))
    binding = {"schema_version": 2, "case_sha256": sha256(case_file), "om_sha256": sha256(om_file),
               "native_kernels": case["native_kernels"], "onnx_sha256": adapter["onnx"]["sha256"],
               "target_soc": adapter["target_soc"], "runtime_dependencies": "device_kernel_only",
               "host_portability": portability, "om_audit_sha256": sha256(audit_path),
               "output_name_bindings": [{"logical_name": "y", "om_declared_names": names}]}
    Path(output).write_text(json.dumps(binding, indent=2) + "\n")


def build_spec(config):
    blocks = config.get("block_dim", 8)
    module = load_kernel(config.get("gym_root"))

    def annotate(manifest):
        manifest["gemm_fused_kernel_configuration"] = {"block_dim": blocks, "negative_slope": 0.01, "dtype": "fp16",
                                                "schedule": "grouped-cube-vector-four-stage"}

    return {"kernels": module.GEMMFusedKernel(blocks), "graph_spec": GRAPH_SPEC, "block_dim": blocks,
            "annotate_manifest": annotate, "make_case": make_case, "bind_case": bind_case}
