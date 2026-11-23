"""FP16 numerical boundaries and the actual Cube/PV plus Vector execution path."""
import itertools

import pytest
import torch

from pypto_1952.isp_1952_validation.gemm_fused_kernel.packing import pack_outputs, pack_weights, unpack_outputs
from pypto_1952.isp_1952_validation.gemm_fused_kernel.provider import load_kernel
from pypto_1952.isp_1952_validation.gemm_fused_kernel.reference import (
    BATCHES, OUTPUT_NAMES, ROW_STARTS, WIDTHS, make_example, postprocess, reference,
)
from pypto_1952.isp_1952_validation.gemm_fused_kernel.validate import run_pipeline


def test_shapes_and_layout():
    inputs, weights = make_example()
    outputs = reference(inputs, weights)
    expected = [(n, 4, 1, 1) if i % 4 == 0 else (n, 4, 4, 1, 1) if i % 4 == 3
                else (n, 1, 1, 1) for i, n in enumerate(BATCHES) for _ in range(2 if i % 4 == 2 else 1)]
    assert len(outputs) == len(OUTPUT_NAMES) == 10
    assert [tuple(x.shape) for x in outputs] == expected
    assert all(x.dtype == torch.float16 for x in outputs)
    packed = pack_outputs(outputs)
    assert sum(x.numel() for x in outputs) == 391
    for want, got in zip(outputs, unpack_outputs(packed)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert got.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert all(x.dtype == torch.float16 for x in pack_weights(weights))


def test_white_balance_range():
    values = torch.tensor(list(itertools.product((-1000., 0., 1000.), repeat=4)), dtype=torch.float16)
    literal = postprocess(values, 0)[0].reshape(-1, 4)
    assert literal.min() > 0.55 and literal.max() < 1.65


def test_heads_are_independent():
    inputs, weights = make_example()
    before = reference(inputs, weights)
    inputs[6] = inputs[6] + 2
    after = reference(inputs, weights)
    for i in (0, 1, 2, 3, 4, 5, 6, 9):
        torch.testing.assert_close(before[i], after[i], rtol=0, atol=0)
    assert not torch.equal(before[7], after[7])
    assert not torch.equal(before[8], after[8])


@pytest.mark.parametrize("blocks", [3, 8])
def test_fp16_cube_vector_pipeline(blocks):
    inputs, weights = make_example(seed=20261111 + blocks)
    actual, stages = run_pipeline(inputs, pack_weights(weights), blocks)
    for tensor in (*inputs, *stages, actual):
        assert tensor.dtype == torch.float16 and torch.isfinite(tensor).all()
    torch.testing.assert_close(actual, pack_outputs(reference(inputs, weights)), atol=0.003, rtol=0.005)


def test_extreme_logits_vector_postprocess():
    _, weights = make_example()
    packed = pack_weights(weights)
    logits = torch.tensor([-1000., 1000., 0., 100.] * 4, dtype=torch.float16).repeat(68, 1)
    expected = torch.zeros_like(logits)
    for head, (start, batch, width) in enumerate(zip(ROW_STARTS, BATCHES, WIDTHS)):
        parts = postprocess(logits[start:start + batch, :width], head % 4)
        expected[start:start + batch, :width] = torch.cat([p.reshape(batch, -1) for p in parts], dim=1)
    actual = torch.full_like(logits, float("nan"))
    load_kernel().GEMMFusedKernel()["postprocess"][None, 8](logits, torch.zeros_like(packed[3]), packed[4], actual)
    assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.005)
    # All 17 color matrices must apply tanh before scaling the signed logits.
    matrices = torch.cat((actual[3:4], actual[52:])).reshape(-1, 4, 4)
    saturated = torch.eye(4, dtype=logits.dtype) + logits[3].sign().reshape(4, 4) * logits.new_tensor(0.15)
    torch.testing.assert_close(matrices, saturated.expand_as(matrices), atol=0, rtol=0)


def test_fp16_leaky_rounding_exact():
    raw = torch.linspace(-4, 4, 68 * 128, dtype=torch.float16).reshape(68, 128)
    raw[0, :4] = raw.new_tensor([-2**-24, 0, 2**-24, -0.125])
    bias = torch.zeros(8, 128, dtype=torch.float16)
    actual = torch.full_like(raw, float("nan"))
    load_kernel().GEMMFusedKernel()["bias_leaky"][None, 8](raw, bias, actual)
    expected = torch.where(raw >= 0, raw, raw * raw.new_tensor(0.01))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_scalar_heads_scale_sigmoid_in_fp16():
    _, weights = make_example()
    packed = pack_weights(weights)
    logits = torch.zeros(68, 16, dtype=torch.float16)
    samples = logits.new_tensor([-1000., -8., -1., 0., 1., 8., 1000., 0.] * 2)
    logits[20:36, 0] = samples
    actual = torch.full_like(logits, float("nan"))
    load_kernel().GEMMFusedKernel()["postprocess"][None, 8](logits, torch.zeros_like(packed[3]), packed[4], actual)
    assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
    assert actual[1, 0].item() == 0.75  # Head 2: sigmoid(0) * 1.5.
    expected = torch.sigmoid(samples) * samples.new_tensor(1.5)
    torch.testing.assert_close(actual[20:36, 0], expected, atol=0.003, rtol=0.005)
    assert actual[23, 0].item() == 0.75  # Head 6's zero-logit sample.
    assert actual[20, 0].item() == 0 and actual[26, 0].item() == 1.5
    assert torch.count_nonzero(actual[1, 1:]) == torch.count_nonzero(actual[20:36, 1:]) == 0
    # Heads 3/7 still use the original unscaled sigmoid for center/contrast.
    tone = logits.new_tensor([0.449951171875, 1.0])
    torch.testing.assert_close(actual[2, :2], tone, atol=0, rtol=0)
    torch.testing.assert_close(actual[36:52, :2], tone.expand(16, 2), atol=0, rtol=0)
    for batch in (1, 16):
        torch.testing.assert_close(postprocess(logits.new_zeros(batch, 1), 1)[0],
                                   logits.new_full((batch, 1, 1, 1), 0.75), atol=0, rtol=0)
