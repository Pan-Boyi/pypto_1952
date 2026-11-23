"""Compiler recipes, golden data and device bridges for the 1952 ISP operators."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSFORMER_ROOT = Path("ops/pypto_pro/experimental/ops_transformer")
GEMM_KERNEL = Path("ops/pro/experimental/gemm_fused_kernel/kernel.py")
