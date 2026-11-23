"""Independent packing, branch, multi-output and static-reduction regressions."""

import copy

import pytest
import torch

from pypto_1952.isp_1952_validation import (
    TRANSFORMER_ROOT, chain_bridges as bridge, chain_golden, chain_layout as layout, chain_provider, golden,
)
from pypto_1952.isp_1952_validation.gemm_fused_kernel import packing
from pypto_1952.provider import load_provider, resolve_gym_root


def test_packed_c1_kernel_c0_weights_and_q16_bias():
    weight = torch.arange(16 * 20 * 9, dtype=torch.float16).reshape(16, 20, 3, 3)
    packed = golden.pack_conv(weight)
    restored = golden.unpack_conv(packed, 3)
    torch.testing.assert_close(restored[:, :20], weight, atol=0, rtol=0)
    assert torch.count_nonzero(restored[:, 20:]) == 0
    bias = golden.q16(torch.tensor([0.125, -0.25, 0.0001]))
    assert bias.dtype == torch.int32
    assert bias.tolist() == [8192, -16384, 7]


def test_dec_enc_local_regions_tails_and_gamma_zero():
    tensors, _, _ = golden.make_case('dec_enc_fused', 31, 113)
    tensors['x'].fill_(1)
    tensors['bias'].zero_()
    weight = torch.zeros(16, 32, 3, 3, dtype=torch.float16)
    weight[0, 4, 1, 1] = 1  # Local scale, channel 0.
    weight[1, 20, 1, 1] = 1  # Local gamma, channel 0.
    tensors['weight'] = golden.pack_conv(weight)
    tensors['tile_scale'][:, 0] = torch.arange(1, 17).view(16, 1, 1)
    tensors['x'][0, 0, 0, 0] = 0
    result = golden.dec_enc_reference(tensors)['output']
    assert tuple(result.shape) == (1, 16, 31, 113)
    assert result[0, 0, 30, 112] == 16
    assert result[0, 0, 0, 29] == 2
    assert result[0, 1, 0, 0] == 1  # 0**0 is one for the prepared zero exponent.
    assert torch.isfinite(result).all()


def test_dec_enc_brightness_uses_image_mean_for_global_and_local():
    tensors = golden.make_inputs('dec_enc_fused', 8, 8)
    x = tensors['x']
    x[:, 0] = 0.375
    x[:, 1, :4] = 0.25
    x[:, 1, 4:] = 0.5
    x[:, 2] = 0.125
    x[:, 3] = 0.625
    tensors['global_bias'][:, 0] = 0.75
    tensors['global_bias'][:, 1] = 0.5
    tensors['tile_bias'][:, 0] = 0.5
    centers = torch.arange(16, dtype=torch.float16) / 64 + 0.5
    tensors['tile_bias'][:, 1, 0, 0] = centers
    tensors['bias'].zero_()
    weight = torch.zeros(16, 32, 3, 3, dtype=torch.float16)
    weight[0, 24, 1, 1] = 1  # Global brightness on channel R.
    weight[1, 30, 1, 1] = 1  # Local brightness on channel G2.
    tensors['weight'] = golden.pack_conv(weight)

    # The two halves of the image have different grayscale, so a tile-local
    # mean or a mean that omits G2 gives a distinct brightness result.
    rgb_planes = x[0, :, (0, 4), 0]
    upper, lower = (0.299 * rgb_planes[0] + 0.587 * rgb_planes[1]
                    + 0.587 * rgb_planes[2] + 0.114 * rgb_planes[3])
    mean = ((upper.float() + lower.float()) / 2).half()
    result = golden.dec_enc_reference(tensors)['output']
    want_global = 0.5 + 0.75 * (x[:, 0] - mean)
    pixel_centers = centers.reshape(4, 4).repeat_interleave(2, 0).repeat_interleave(2, 1)
    want_local = pixel_centers + 0.5 * (x[:, 2] - mean)
    torch.testing.assert_close(result[:, 0], want_global, atol=0, rtol=0)
    torch.testing.assert_close(result[:, 1], want_local, atol=0, rtol=0)


@pytest.mark.parametrize('height,width', [(31, 113), (1080, 1920)])
def test_dec_enc_mean_nodes_match_pytorch_fp16(height, width):
    _, module = load_provider(resolve_gym_root() / TRANSFORMER_ROOT / 'dec_enc_fused/brightness.py')
    partial_kernel = copy.copy(module.gray_partials_kernel)
    mean_kernel = copy.copy(module.image_mean_kernel)
    for kernel in (partial_kernel, mean_kernel):
        kernel._backend = 'cpu'
        kernel._compiled_by_signature = {}
    generator = torch.Generator().manual_seed(2711)
    x = torch.rand((1, 4, height, width), generator=generator).half()
    x[:, 1] *= 0.625
    x[:, 2] *= 0.375
    partials = torch.full((8, 8), float('nan'), dtype=torch.float32)
    mean = torch.full((1, 16), float('nan'), dtype=torch.float16)
    partial_kernel[None, 8](x, partials)
    mean_kernel[None, 8](x, partials, mean)
    assert torch.isfinite(partials).all()
    assert torch.count_nonzero(partials[:, 1:]) == 0
    torch.testing.assert_close(mean[0, 0], golden.weighted_image_mean(x).squeeze(), atol=0, rtol=0)


def test_preprocess_neutral_chroma_replicates_edges_and_gathers_lut():
    x = torch.tensor([[[[64., 128.]], [[192., 256.]], [[2048., 2048.]], [[2048., 2048.]]]],
                     dtype=torch.float16)
    lut = torch.arange(4096).half() * 0.5
    expected = torch.tensor([[[[128., 176.]], [[64., 128.]], [[192., 256.]], [[144., 192.]]]],
                            dtype=torch.float16) / 8192
    torch.testing.assert_close(golden.preprocess_reference({'x': x, 'lut': lut})['out'],
                               expected, atol=0, rtol=0)


def test_ffm_projects_raw_head_and_four_independent_y_channels():
    tensors = {'x': torch.zeros(1, 16, 4, 4, dtype=torch.float16)}
    for layer, kernel in ((1, 3), (2, 3), (3, 1)):
        tensors['w' + str(layer)] = golden.pack_conv(torch.zeros(16, 16, kernel, kernel, dtype=torch.float16))
        tensors['b' + str(layer)] = torch.zeros(1, 16, dtype=torch.int32)
    tensors['b3'][0, :4] = golden.q16(torch.tensor([0.125, 0.25, 0.75, 0.5]))
    tensors['coeff'] = torch.zeros(1, 32)
    tensors['coeff'][0, :24] = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.],
                                           [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 0.]]).flatten()
    tensors['cb'] = torch.tensor([[0., 0., 0.25, 0., 0., 2., 0., 0.]])
    result = golden.ffm_reference(tensors)
    expected = torch.tensor([0.125, 0.5, 0.25, 0.5], dtype=torch.float16)
    torch.testing.assert_close(result['yout'][0, :, 0, 0], expected, atol=0, rtol=0)
    assert result['uvout'][0, 0, 0, 0] == 0.5
    assert result['uvout'][0, 1, 0, 0] == 1


def test_bridges_preserve_feature_rows_and_fp16_coefficients():
    features = torch.arange(17, dtype=torch.float16).view(17, 1).repeat(1, 128)
    heads = layout.head_inputs(features.numpy().tobytes())
    assert all(heads['x' + str(i)] == features[:1].numpy().tobytes() for i in range(4))
    assert all(heads['x' + str(i)] == features[1:].numpy().tobytes() for i in range(4, 8))
    global_features = torch.full_like(features[:1], float('nan'))
    local_features = torch.full_like(features[1:], float('nan'))
    bridge.split_features[None, 1](features, global_features, local_features)
    assert global_features.numpy().tobytes() == heads['x0']
    assert local_features.numpy().tobytes() == heads['x4']

    prediction = []
    for batch in (1, 16):
        region = torch.arange(batch, dtype=torch.float16).reshape(batch, 1, 1, 1)
        prediction.extend((region.expand(batch, 4, 1, 1) + 1, region / 32 + 0.75,
                           region / 64 + 0.25, region / 32 + 0.75,
                           region.unsqueeze(2).expand(batch, 4, 4, 1, 1)))
    packed = packing.pack_outputs(prediction)
    coefficients = chain_golden.prepare_dec_coefficients(prediction)
    assert coefficients['global_bias'].flatten().tolist() == [0.75, 0.25]
    assert coefficients['tile_bias'][-1, :, 0, 0].tolist() == [1.21875, 0.484375]
    expected = layout.dec_coefficients(packed.numpy().tobytes())
    assert expected == {n: p.numpy().tobytes() for n, p in coefficients.items()}
    actual = {name: torch.full(shape, float('nan'), dtype=torch.float16)
              for name, shape in layout.DEC_COEFF_SHAPES.items()}
    bridge.unpack_coefficients[None, 1](packed, *actual.values())
    assert {n: p.numpy().tobytes() for n, p in actual.items()} == expected


def test_single_om_graph_covers_every_inclusive_span():
    counts = dict(zip(layout.STAGES, (1, 3, 4, 3, 1, 1)))
    for first, start in enumerate(layout.STAGES):
        for last, end in enumerate(layout.STAGES[first:], first):
            span = layout.STAGES[first:last + 1]
            spec = chain_provider.build_spec({'start': start, 'end': end})
            graph = spec['graph_spec']
            nodes = {node['name']: node for node in graph['nodes']}
            split = {'input_fused', 'GEMMFusedKernel'} <= set(span)
            unpack = {'GEMMFusedKernel', 'dec_enc_fused'} <= set(span)
            assert ('split_features' in nodes) == split
            assert ('unpack_coefficients' in nodes) == unpack
            assert len(nodes) == len(spec['kernels']) == sum(counts[op] for op in span) + split + unpack

            available = set(graph['inputs'])
            for node in graph['nodes']:
                assert set(node['inputs']) <= available
                assert not (set(node['outputs']) & available)
                available.update(node['outputs'])
            assert set(graph['outputs']) <= available
            assert all(name.startswith(end + '_') for name in graph['outputs'])
            assert len(graph['outputs']) == (2 if end == 'ffm_fused' else 1)
            if split:
                assert nodes['GEMMFusedKernel_fc1']['inputs'][:8] == (
                    ['global_features'] * 4 + ['local_features'] * 4)
            if unpack:
                assert nodes['dec_enc_fused_dec_enc_fused']['inputs'][1:9] == nodes['unpack_coefficients']['outputs']
            if {'input_fused', 'dec_enc_fused'} <= set(span):
                image = 'preprocess_out' if start == 'preprocess' else 'input_fused_x'
                assert nodes['input_fused_resize']['inputs'][0] == image
                assert nodes['dec_enc_fused_dec_enc_fused']['inputs'][0] == image


def test_static_reducer_preserves_all_local_rows():
    from pypto_pro._errors import NotSupported
    _, module = load_provider(resolve_gym_root() / TRANSFORMER_ROOT / 'input_fused/input_fused.py')
    kernel = copy.copy(module.reduce_kernel)
    kernel._backend = 'cpu'
    kernel._compiled_by_signature = {}
    local = torch.arange(17 * 128, dtype=torch.float16).reshape(17, 128)
    partials = (torch.arange(8 * 128, dtype=torch.int32).reshape(8, 128) - 512) * 64
    actual = torch.full_like(local, float('nan'))
    try:
        kernel[None, 8](local, partials, actual)
    except NotSupported as error:
        if any(message in str(error) for message in (
                'CPU VF currently supports FP16 and FP32 registers only',
                'CPU tensor views do not support dtype reinterpretation')):
            pytest.skip('CPU emitter lacks INT32 bit-transport support; run the native reducer test on board')
        raise
    torch.testing.assert_close(actual[1:], local[1:], atol=0, rtol=0)
    expected = (partials.float().sum(0) / 65536).half()
    torch.testing.assert_close(actual[0], expected, atol=0, rtol=0)
