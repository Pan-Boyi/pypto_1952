"""Generate CPU golden data for contiguous ranges of the FP16 ISP graph."""

import torch

from . import chain_layout as LAYOUT, golden as GOLDEN
from .gemm_fused_kernel import packing, reference as ref
from .provider import write_case


def prepare_dec_coefficients(prediction):
    """Pass the GEMM predictions as (contrast, center)."""
    gwb, gexp, gcenter, gcontrast, gmix, lwb, lexp, lcenter, lcontrast, lmix = prediction
    coeff = {
        'channel_scale': gwb,
        'global_exponent': gexp,
        'global_bias': torch.cat((gcontrast, gcenter), dim=1),
        'channel_mix': gmix,
        'tile_scale': lwb,
        'tile_exponent': lexp,
        'tile_bias': torch.cat((lcontrast, lcenter), dim=1),
        'tile_mix': lmix,
    }
    for name, value in coeff.items():
        if value.dtype != torch.float16 or tuple(value.shape) != LAYOUT.DEC_COEFF_SHAPES[name]:
            raise ValueError('GEMMFusedKernel prediction does not match DecEnc input: ' + name)
    return {name: value.contiguous() for name, value in coeff.items()}


def iter_cases(seed):
    """Evaluate each stage with the actual CPU outputs of its predecessors."""
    _, _, height, width = LAYOUT.IMAGE_SHAPE
    x = GOLDEN.make_inputs('preprocess', height, width, seed)
    pre = GOLDEN.preprocess_reference(x)
    yield 'preprocess', x, pre
    image = pre['out']
    x = GOLDEN.make_inputs('input_fused', height, width, seed + 1, image=image)
    features = GOLDEN.input_reference(x)[0]
    yield 'input_fused', x, features
    features = features['features']

    _, weights = ref.make_example(seed=seed + 2)
    heads = [features[:1]] * 4 + [features[1:]] * 4
    prediction = ref.reference(heads, weights)
    yield 'GEMMFusedKernel', dict(zip(packing.INPUT_NAMES, heads + list(packing.pack_weights(weights)))), {
        'y': packing.pack_outputs(prediction),
    }

    x = GOLDEN.make_inputs('dec_enc_fused', height, width, seed + 3, image=image)
    x.update(prepare_dec_coefficients(prediction))
    encoded = GOLDEN.dec_enc_reference(x)
    yield 'dec_enc_fused', x, encoded
    x = GOLDEN.make_inputs('ffm_fused', height, width, seed + 4, image=encoded['output'])
    ffm = GOLDEN.ffm_reference(x)
    yield 'ffm_fused', x, ffm
    x = {'y4': ffm['yout'], 'uv': ffm['uvout']}
    yield 'postprocess', x, GOLDEN.postprocess_reference(x)


def make_span_case(output, seed, stages, external_inputs, model_outputs):
    """Save one fixture containing the OM boundary tensors and final reference."""
    collected = {}
    for operator, inputs, outputs in iter_cases(seed):
        collected.update((name, inputs[port]) for name, (stage, port) in external_inputs.items()
                         if stage == operator)
        if operator == stages[-1]:
            reference = {name: outputs[port] for name, port in model_outputs.items()}
            break

    if any(value.dtype != torch.float16 or not value.isfinite().all() for value in reference.values()):
        raise ValueError('Invalid ISP golden output: ' + stages[-1])
    _, _, height, width = LAYOUT.IMAGE_SHAPE
    config = {'operator': 'ISPChain', 'height': height, 'width': width, 'seed': seed,
              'fixture': 'isp_chain_single_om_fp16', 'brightness_rule': LAYOUT.BRIGHTNESS_RULE,
              'start': stages[0], 'end': stages[-1], 'stages': list(stages)}
    # NV12 rearranges the preceding Y/UV values. A span ending in Postprocess
    # inherits FFM's tolerance when it includes FFM; standalone packing is exact.
    tolerance = 'ffm_fused' if stages[-1] == 'postprocess' and 'ffm_fused' in stages else stages[-1]
    boundary = {name: collected[name] for name in external_inputs}
    return write_case(output, boundary, reference, GOLDEN.TOLERANCES[tolerance], config)
