"""FP16 ISP graph order, links and reference layouts."""

STAGES = ('preprocess', 'input_fused', 'GEMMFusedKernel', 'dec_enc_fused', 'ffm_fused', 'postprocess')
IMAGE_SHAPE = (1, 4, 1080, 1920)
FEATURES_SHAPE = (17, 128)
GEMM_OUTPUT_SHAPE = (68, 16)
BRIGHTNESS_RULE = 'center_plus_contrast_times_input_minus_weighted_image_mean_fp16'
DIRECT_INPUTS = {
    'input_fused': {'x': ('preprocess', 'out')},
    'dec_enc_fused': {'x': ('preprocess', 'out')},
    'ffm_fused': {'x': ('dec_enc_fused', 'output')},
    'postprocess': {'y4': ('ffm_fused', 'yout'), 'uv': ('ffm_fused', 'uvout')},
}
DEC_COEFF_SHAPES = {
    'channel_scale': (1, 4, 1, 1),
    'global_exponent': (1, 1, 1, 1),
    'global_bias': (1, 2, 1, 1),
    'channel_mix': (1, 4, 4, 1, 1),
    'tile_scale': (16, 4, 1, 1),
    'tile_exponent': (16, 1, 1, 1),
    'tile_bias': (16, 2, 1, 1),
    'tile_mix': (16, 4, 4, 1, 1),
}


def selected_stages(start=None, end=None):
    start = STAGES[0] if start is None else start
    end = STAGES[-1] if end is None else end
    if start not in STAGES or end not in STAGES:
        raise ValueError('Choose --start and --end from ' + ', '.join(STAGES))
    first, last = STAGES.index(start), STAGES.index(end)
    if first > last:
        raise ValueError('--start must be at or before --end')
    return STAGES[first:last + 1]


def head_inputs(features):
    """Split little-endian FP16 features and reuse each group for four heads."""
    if len(features) != 17 * 128 * 2:
        raise ValueError('Expected packed FP16 [17,128] InputFused features')
    return dict(zip(('x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'),
                    (features[:256],) * 4 + (features[256:],) * 4))


def dec_coefficients(values):
    """Read GEMM's FP16 [68,16] output and produce eight dense FP16 inputs."""
    if len(values) != 68 * 16 * 2:
        raise ValueError('Expected packed FP16 [68,16] GEMMFusedKernel output')

    def rows(start, count, width):
        return b''.join(values[row * 32:row * 32 + width * 2] for row in range(start, start + count))

    def tone(start, count):
        # GEMM packs (center, contrast), DecEnc accepts (contrast, center).
        packed = rows(start, count, 2)
        return b''.join(packed[i + 2:i + 4] + packed[i:i + 2] for i in range(0, len(packed), 4))

    return {
        'channel_scale': rows(0, 1, 4),
        'global_exponent': rows(1, 1, 1),
        'global_bias': tone(2, 1),
        'channel_mix': rows(3, 1, 16),
        'tile_scale': rows(4, 16, 4),
        'tile_exponent': rows(20, 16, 1),
        'tile_bias': tone(36, 16),
        'tile_mix': rows(52, 16, 16),
    }
