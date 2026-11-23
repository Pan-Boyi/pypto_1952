"""Prepare weights once; the hot path accepts the eight original input buffers directly."""

import torch

from .reference import BATCHES, ROW_STARTS, WIDTHS, validate

INPUT_NAMES = tuple(f"x{i}" for i in range(8)) + ("w1t", "b1", "w2t", "b2", "constants")
SHAPES = tuple((n, 128) for n in BATCHES) + ((8, 128, 128), (8, 128), (8, 128, 16), (8, 128), (7, 128))


def pack_weights(weights):
    """FP16 [K,N] weights are consumed directly by the native Cube GEMMs."""
    validate([w.w1.new_empty(n, 128) for n, w in zip(BATCHES, weights)], weights)
    device = weights[0].w1.device
    w1t = torch.stack([w.w1.half().T.contiguous() for w in weights]).contiguous()
    b1 = torch.stack([w.b1.half() for w in weights]).contiguous()
    w2t = torch.zeros(8, 128, 16, dtype=torch.float16, device=device)
    b2 = torch.zeros(8, 128, dtype=torch.float16, device=device)
    for head, (w, width) in enumerate(zip(weights, WIDTHS)):
        w2t[head, :, :width] = w.w2.half().T
        b2[head, :width] = w.b2.half()
    constants = torch.zeros(7, 128, dtype=torch.float16, device=device)
    base = constants.new_tensor([470., 512., 512., 515.])
    constants[0, :4] = base / base.mean()  # Preserve the original FP16 rounding, fold once.
    constants[1, :16] = torch.eye(4, device=device).flatten()
    for row, width in enumerate((4, 1, 2, 16), start=2):
        constants[row, :width] = 1
    constants[6, 1] = 1
    return w1t, b1, w2t, b2, constants


def pack_outputs(outputs):
    """Single-output OM ABI, zero-filled padding; meaningful values total 391 FP16s."""
    y = outputs[0].new_zeros(68, 16)
    index = 0
    for head, (batch, width, start) in enumerate(zip(BATCHES, WIDTHS, ROW_STARTS)):
        if head % 4 == 2:
            y[start:start + batch, :2] = torch.cat([outputs[index].reshape(batch, 1),
                                                  outputs[index + 1].reshape(batch, 1)], dim=1)
            index += 2
        else:
            y[start:start + batch, :width] = outputs[index].reshape(batch, width)
            index += 1
    return y


def unpack_outputs(y):
    """Ten strided views. No device math, copy, or extra kernel is necessary."""
    if tuple(y.shape) != (68, 16):
        raise ValueError("Expected the packed output shape [68,16]")
    results = []
    for head, (batch, width, start) in enumerate(zip(BATCHES, WIDTHS, ROW_STARTS)):
        p = y[start:start + batch]
        if head % 4 == 2:
            results.extend((p[:, :1].view(batch, 1, 1, 1), p[:, 1:2].view(batch, 1, 1, 1)))
        else:
            shape = (batch, 4, 4, 1, 1) if width == 16 else (batch, width, 1, 1)
            results.append(p[:, :width].view(shape))
    return tuple(results)
