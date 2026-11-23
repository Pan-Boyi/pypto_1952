"""Independent CPU oracles and deterministic inputs for the FP16 ISP path."""

import torch
import torch.nn.functional as F

from .reference_nv12 import to_yuv420_nv12

TOLERANCES = {
    'preprocess': (0.0, 0.0),
    'input_fused': (0.003, 0.005),
    'GEMMFusedKernel': (0.003, 0.005),
    'dec_enc_fused': (0.0005, 0.001),
    'ffm_fused': (0.003, 0.003),
    'postprocess': (0.0, 0.0),
}


def preprocess_reference(t):
    """Inverse Star-Tetrix with replicate edges, stepwise FP16 and a clamped LUT."""
    y1, y2, cb, cr = t['x'].unbind(1)
    cb = cb - 2048
    cr = cr - 2048

    def neighbor(plane, axis, step):
        indices = (torch.arange(plane.shape[axis]) + step).clamp(0, plane.shape[axis] - 1)
        return plane.index_select(axis, indices)

    g1 = y1 - ((cr + neighbor(cr, 2, 1) + cb + neighbor(cb, 1, -1)) * 0.25).floor()
    g2 = y2 - ((neighbor(cb, 2, -1) + cb + neighbor(cr, 1, 1) + cr) * 0.25).floor()
    red = 2 * cr + ((neighbor(g1, 2, -1) + g1 + neighbor(g2, 1, -1) + g2) * 0.25).floor()
    blue = 2 * cb + ((g2 + neighbor(g2, 2, 1) + g1 + neighbor(g1, 1, 1)) * 0.25).floor()
    index = torch.stack((red, g1, g2, blue), dim=1).long().clamp(0, 4095)
    return {'out': t['lut'][index] * (1 / 4096)}


def pack_conv(weight):
    out, channels, kh, kw = weight.shape
    padded = weight.new_zeros(out, (channels + 15) // 16 * 16, kh, kw)
    padded[:, :channels] = weight
    return padded.reshape(out, -1, 16, kh, kw).permute(1, 3, 4, 2, 0).reshape(-1, out).contiguous()


def unpack_conv(weight, kernel):
    channels = weight.shape[0] // (kernel * kernel)
    return weight.reshape(channels // 16, kernel, kernel, 16, -1).permute(4, 0, 3, 1, 2).reshape(-1, channels, kernel, kernel)


def q16(bias):
    return torch.round(bias.float() * 65536).to(torch.int32)


def convolution(x, weight, bias, kernel, leaky=True):
    w = unpack_conv(weight, kernel).float()
    x = x.float()
    if x.shape[1] < w.shape[1]:
        x = F.pad(x, (0, 0, 0, 0, 0, w.shape[1] - x.shape[1]))
    y = F.conv2d(x, w, bias.float().flatten() / 65536, padding=kernel // 2)
    return (F.leaky_relu(y, 0.01) if leaky else y).half()


def weighted_image_mean(x):
    """Compute one FP16 mean per NCHW image, with FP32 reduction accumulation."""
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.587 * x[:, 2:3] + 0.114 * x[:, 3:4]
    return gray.mean(dim=(2, 3), keepdim=True)


def dec_enc_reference(t):
    x = t['x']
    _, _, height, width = x.shape
    local_index = ((torch.arange(height) // ((height + 3) // 4))[:, None] * 4
                   + (torch.arange(width) // ((width + 3) // 4))[None, :])

    def local(name):
        return t[name].reshape(16, -1)[local_index].permute(2, 0, 1).unsqueeze(0)

    def mix(coeff):
        # The kernel rounds the first product and each subsequent fused AXPY to FP16.
        value = x[:, :1] * coeff[:, :, 0]
        for channel in range(1, 4):
            value = torch.addcmul(value, x[:, channel:channel + 1], coeff[:, :, channel])
        return value

    def gamma(exponent):
        return torch.where(x > 0, (x.log() * exponent).exp(), (exponent == 0).to(x.dtype))

    global_mix = mix(t['channel_mix'])
    local_mix = mix(local('tile_mix').reshape(1, 4, 4, height, width))
    # GEMM supplies (contrast, center) in both tables. Both use the same image mean.
    gb, lb = t['global_bias'].reshape(1, 2, 1, 1), local('tile_bias')
    centered = x - weighted_image_mean(x)
    branches = [x * t['channel_scale'], x * local('tile_scale'), global_mix, local_mix,
                gamma(t['global_exponent']), gamma(local('tile_exponent')),
                gb[:, 1:2] + gb[:, :1] * centered, lb[:, 1:2] + lb[:, :1] * centered]
    prepared = torch.cat(branches, dim=1)
    return {'output': convolution(prepared, t['weight'], t['bias'], 3)}


def ffm_reference(t):
    y = convolution(t['x'], t['w1'], t['b1'], 3)
    y = convolution(y, t['w2'], t['b2'], 3)
    y = convolution(y, t['w3'], t['b3'], 1, leaky=False)
    # The current kernel projects raw head output. Each product/add rounds in FP16.
    green = y[:, 1:2] * 0.5 + y[:, 2:3] * 0.5
    colors = (y[:, :1], green, green, y[:, 3:4])
    weights, bias = t['coeff'].flatten().half(), t['cb'].flatten().half()
    channels = []
    for channel in range(6):
        value = colors[0] * weights[channel * 4]
        for component in range(1, 4):
            value = value + colors[component] * weights[channel * 4 + component]
        channels.append((value + bias[channel]).clamp(0, 1))
    return {'yout': torch.cat(channels[:4], 1), 'uvout': torch.cat(channels[4:], 1)}


def input_reference(t):
    x = t['x'].float()
    tile_h, tile_w = x.shape[2] // 4, x.shape[3] // 4
    views = [x] + [x[:, :, row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w]
                   for row in range(4) for col in range(4)]
    resized = torch.cat([F.interpolate(view, size=(128, 128), mode='bilinear', align_corners=False)
                         for view in views]).half()
    outputs = []
    for bank, samples in ((0, resized[:1]), (1, resized[1:])):
        y = samples
        for layer, kernel in enumerate((7, 5, 3), 1):
            y = convolution(y, t['w' + str(layer)][bank], t['b' + str(layer)][bank], kernel)
            y = F.avg_pool2d(y.float(), 2).half()
        outputs.append(y.float().mean(dim=(2, 3)).half())
    return {'features': torch.cat(outputs)}, resized


def postprocess_reference(t):
    return {'out': to_yuv420_nv12(t['y4'], t['uv'])}


REFERENCES = {
    'preprocess': preprocess_reference,
    'input_fused': lambda t: input_reference(t)[0],
    'dec_enc_fused': dec_enc_reference,
    'ffm_fused': ffm_reference,
    'postprocess': postprocess_reference,
}


def make_inputs(operator, height=1080, width=1920, seed=20261117, image=None):
    generator = torch.Generator().manual_seed(seed)

    def random(shape, scale=1.0, offset=0.0, half=False):
        value = torch.randn(shape, generator=generator) * scale + offset
        return value.half() if half else value

    def weight(out, channels, kernel):
        return pack_conv(random((out, channels, kernel, kernel), 0.015, half=True))

    if operator == 'preprocess':
        if height < 1 or width != 1920:
            raise ValueError('Preprocess requires positive height and width 1920')
        x = torch.randint(0, 4096, (1, 4, height, width), generator=generator).half()
        return {'x': x, 'lut': torch.arange(4096, dtype=torch.int32).half()}
    if operator == 'postprocess':
        return {'y4': random((1, 4, height, width), 0.45, 0.5, half=True),
                'uv': random((1, 2, height, width), 0.45, 0.5, half=True)}
    if operator == 'dec_enc_fused':
        if min(height, width) < 4:
            raise ValueError('DecEncFused requires height and width >= 4')
        if image is None:
            image = torch.rand((1, 4, height, width), generator=generator).half()
            image[:, :, 0, 0] = 0
        return {
            'x': image,
            'channel_scale': random((1, 4, 1, 1), 0.05, 1, half=True),
            'global_exponent': torch.tensor([0.8], dtype=torch.float16).reshape(1, 1, 1, 1),
            'global_bias': torch.tensor([1.1, 0.45], dtype=torch.float16).reshape(1, 2, 1, 1),
            'channel_mix': (torch.eye(4) + random((4, 4), 0.02)).half().reshape(1, 4, 4, 1, 1),
            'tile_scale': random((16, 4, 1, 1), 0.1, 1, half=True),
            'tile_exponent': torch.linspace(0, 1.5, 16).half().reshape(16, 1, 1, 1),
            'tile_bias': torch.stack((torch.linspace(0.8, 1.2, 16), torch.linspace(0.25, 0.65, 16)), 1)
            .half().reshape(16, 2, 1, 1),
            'tile_mix': (torch.eye(4)[None] + random((16, 4, 4), 0.03)).half().reshape(16, 4, 4, 1, 1),
            'weight': weight(16, 32, 3), 'bias': q16(random((1, 16), 0.02)),
        }
    if (height, width) != (1080, 1920):
        raise ValueError(operator + ' canonical kernel requires height=1080, width=1920')
    if operator == 'ffm_fused':
        return {'x': image if image is not None else random((1, 16, height, width), 0.2, half=True),
             'w1': weight(16, 16, 3), 'b1': q16(random((1, 16), 0.02)),
             'w2': weight(16, 16, 3), 'b2': q16(random((1, 16), 0.02)),
             'zero': torch.zeros(1920, 16, dtype=torch.float16),
             'w3': weight(16, 16, 1), 'b3': q16(random((1, 16), 0.02)),
             'coeff': random((1, 32), 0.15, 0.2), 'cb': random((1, 8), 0.05)}
    if operator != 'input_fused':
        raise ValueError('Unknown transformer operator: ' + operator)
    t = {'x': image if image is not None else torch.rand((1, 4, height, width), generator=generator).half()}
    for layer, (channels, out, kernel) in enumerate(((4, 16, 7), (16, 32, 5), (32, 128, 3)), 1):
        t['w' + str(layer)] = torch.stack([weight(out, channels, kernel) for _ in range(2)])
        t['b' + str(layer)] = q16(random((2, out), 0.02))
    t['zeros'] = torch.zeros(128, 16, dtype=torch.float16)
    t['poolweights'] = torch.eye(16, dtype=torch.float16).repeat(4, 1) * 0.25
    t['ones'] = torch.zeros(16, 16, dtype=torch.float16)
    t['ones'][0] = 1 / 256
    return t


def make_case(operator, height=1080, width=1920, seed=20261117):
    inputs = make_inputs(operator, height, width, seed)
    return inputs, REFERENCES[operator](inputs), TOLERANCES[operator]
