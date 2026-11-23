#!/usr/bin/env bash
# Compiler-side entry point. Uses the caller's Python/conda/venv; never activates an environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
deploy_root=$(cd -- "$script_dir/.." && pwd)
gym_root=${PYPTO_GYM_ROOT:-"$deploy_root/../pypto-gym"}
provider="$deploy_root/pypto_1952/isp_1952_validation/gemm_fused_kernel/provider.py"
phase=${1:---help}
if [[ $# -gt 0 ]]; then shift; fi
case "$phase" in
    build)
        options=(--provider "$provider" --gym-root "$gym_root" --op GEMMFusedKernel --dtype fp16 --block-dim 8)
        toolkit=${PYPTO_PRO_TOOLKIT_ROOT:-${CANN_SDK_ROOT:-}}
        if [[ -n "$toolkit" ]]; then options+=(--toolkit-root "$toolkit"); fi
        if [[ -n "${PTO_LIB_PATH:-}" ]]; then options+=(--pto-root "$PTO_LIB_PATH"); fi
        exec bash "$deploy_root/scripts/static_1952.sh" build "${options[@]}" "$@"
        ;;
    package|accept)
        exec bash "$deploy_root/scripts/static_1952.sh" "$phase" "$@"
        ;;
    -h|--help)
        cat <<'HELP'
FP16 GEMMFusedKernel: compile on A, transfer runtime.tar.gz to B, return results.tar.gz to A.
  test_gemm_fused_kernel.sh build --output /new/build [--gym-root GYM --toolkit-root CANN --pto-root PTO]
  test_gemm_fused_kernel.sh package --build /existing/build --output /new/package
  test_gemm_fused_kernel.sh accept --build /existing/build --results-archive /returned/results.tar.gz --output /new/check
Build options are forwarded to pypto_1952; use build --help for the complete list.
PYTHON selects the compiler interpreter. No SSH, NPU, or automatic venv activation is required.
HELP
        ;;
    *) echo "Unknown phase: $phase (use --help)" >&2; exit 2 ;;
esac
