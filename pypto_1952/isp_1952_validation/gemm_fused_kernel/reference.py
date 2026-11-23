"""Readable PyTorch oracle. Weight convention is nn.Linear [out_features, in_features]."""

from dataclasses import dataclass

import torch

BATCHES = (1, 1, 1, 1, 16, 16, 16, 16)
WIDTHS = (4, 1, 2, 16, 4, 1, 2, 16)
ROW_STARTS = (0, 1, 2, 3, 4, 20, 36, 52)
OUTPUT_NAMES = ("global_wb", "global_scalar", "global_center", "global_contrast", "global_matrix",
                "local_wb", "local_scalar", "local_center", "local_contrast", "local_matrix")


@dataclass
class HeadWeights:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


def postprocess(delta, kind):
    """Literal requested operations; second WB normalization means mean(dim=1)."""
    batch = delta.shape[0]
    if kind == 0:
        delta = delta.reshape(batch, 4, 1, 1)
        base = delta.new_tensor([470.0, 512.0, 512.0, 515.0]).view(1, 4, 1, 1)
        base = base / base.mean(dim=1, keepdim=True).clamp_min(1e-6)
        params = base * (delta.new_tensor(1.0) + delta.new_tensor(0.15) * torch.tanh(delta))
        params = params / params.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return (params.clamp(0.55, 1.65),)
    if kind == 1:
        return ((torch.sigmoid(delta) * delta.new_tensor(1.5)).reshape(batch, 1, 1, 1),)
    if kind == 2:
        p = torch.sigmoid(delta)
        center = (p[:, :1] * p.new_tensor(0.5) + p.new_tensor(0.2)).reshape(batch, 1, 1, 1)
        contrast = torch.exp(p.new_tensor(0.3) * (p.new_tensor(2.0) * p[:, 1:2] - p.new_tensor(1.0))).reshape(batch, 1, 1, 1)
        return center, contrast
    identity = torch.eye(4, dtype=delta.dtype, device=delta.device)
    adjustment = torch.tanh(delta).reshape(batch, 4, 4) * delta.new_tensor(0.15)
    return ((identity + adjustment).reshape(batch, 4, 4, 1, 1),)


def reference(inputs, weights, negative_slope=0.01):
    """Return ten tensors in OUTPUT_NAMES order; all eight parameter sets are independent."""
    validate(inputs, weights)
    results = []
    for head, (x, weight) in enumerate(zip(inputs, weights)):
        # Separate FP16 boundaries for GEMM, bias and activation, matching the device graph.
        hidden = x @ weight.w1.T
        hidden = hidden + weight.b1
        hidden = torch.where(hidden >= 0, hidden, hidden * hidden.new_tensor(negative_slope))
        delta = hidden @ weight.w2.T
        delta = delta + weight.b2
        results.extend(postprocess(delta, head % 4))
    return tuple(results)


def validate(inputs, weights):
    if len(inputs) != 8 or len(weights) != 8:
        raise ValueError("Exactly eight inputs and eight independent weight sets are required")
    for head, (x, w, batch, width) in enumerate(zip(inputs, weights, BATCHES, WIDTHS)):
        shapes = (tuple(x.shape), tuple(w.w1.shape), tuple(w.b1.shape), tuple(w.w2.shape), tuple(w.b2.shape))
        if shapes != ((batch, 128), (128, 128), (128,), (width, 128), (width,)):
            raise ValueError(f"Head {head + 1} has incorrect shapes: {shapes}")


def make_example(seed=20261111, dtype=torch.float16):
    rng = torch.Generator().manual_seed(seed)
    inputs = [torch.randn(n, 128, generator=rng, dtype=dtype) * 0.5 for n in BATCHES]
    weights = [HeadWeights(torch.randn(128, 128, generator=rng, dtype=dtype) / 16,
                           torch.randn(128, generator=rng, dtype=dtype) / 8,
                           torch.randn(c, 128, generator=rng, dtype=dtype) / 16,
                           torch.randn(c, generator=rng, dtype=dtype) / 8) for c in WIDTHS]
    return inputs, weights


if __name__ == "__main__":
    xs, ws = make_example()
    for name, result in zip(OUTPUT_NAMES, reference(xs, ws)):
        print(name, tuple(result.shape), result.flatten()[:4].tolist())
