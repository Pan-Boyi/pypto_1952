"""Record FP16 pipeline errors; CPU and Cube/PV are not board performance results."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from pypto_1952.isp_1952_validation import GEMM_KERNEL
from pypto_1952.provider import resolve_gym_root

from .packing import pack_outputs, pack_weights, unpack_outputs
from .provider import load_kernel
from .reference import OUTPUT_NAMES, make_example, reference


def run_pipeline(inputs, packed, blocks=8, gym_root=None):
    kernels = load_kernel(gym_root).GEMMFusedKernel(blocks)
    raw = torch.full((68, 128), float("nan"), dtype=torch.float16)
    hidden = torch.full_like(raw, float("nan"))
    delta = torch.full((68, 16), float("nan"), dtype=torch.float16)
    y = torch.full_like(delta, float("nan"))
    w1, b1, w2, b2, const = packed
    kernels["first_gemm"][None, blocks](*inputs, w1, raw)
    kernels["bias_leaky"][None, blocks](raw, b1, hidden)
    kernels["second_gemm"][None, blocks](hidden, w2, delta)
    kernels["postprocess"][None, blocks](delta, b2, const, y)
    return y, (raw, hidden, delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, weights = make_example()
    expected = reference(inputs, weights)
    gym = resolve_gym_root(args.gym_root)
    actual, stages = run_pipeline(inputs, pack_weights(weights), gym_root=gym)
    torch.testing.assert_close(actual, pack_outputs(expected), atol=0.003, rtol=0.005)
    metrics = {}
    for name, want, got in zip(OUTPUT_NAMES, expected, unpack_outputs(actual)):
        error = (got.double() - want.double()).abs()
        metrics[name] = {"shape": list(got.shape), "dtype": str(got.dtype),
                         "max_absolute_error": error.max().item(), "mean_absolute_error": error.mean().item()}
    root = Path(__file__).parent
    report = {"status": "fp16-cpu-cube-pv-numerical-passed", "device_execution": "not_run",
              "device_performance": "not_measured", "cube_instruction_pv_execution": "passed",
              "full_native_program_pv_execution": "not_run", "seed": 20261111, "atol": 0.003, "rtol": 0.005,
              "block_dim": 8, "tensor_dtype": "fp16", "accumulator": "native INT32 fixed-point",
              "intermediates": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                  "finite": bool(torch.isfinite(value).all())}
                                 for name, value in zip(("raw", "hidden", "delta"), stages)],
              "outputs": metrics,
              "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted(root.glob("*.py")), gym / GEMM_KERNEL]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
