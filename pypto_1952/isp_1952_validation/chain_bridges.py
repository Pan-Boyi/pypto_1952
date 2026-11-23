"""Device kernels for the two FP16 layout changes in the ISP graph."""

import pypto_pro.language as pl


@pl.jit(backend='cpu', arch='5102', auto_mutex=False, name='isp_split_features_5102')
def split_features(
    features: pl.Tensor[[17, 128], pl.DT_FP16],
    global_features: pl.Tensor[[1, 128], pl.DT_FP16, pl.Output],
    local_features: pl.Tensor[[16, 128], pl.DT_FP16, pl.Output],
):
    with pl.section_vector():
        global_tile = pl.make_tile(
            pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0)
        local_tile = pl.make_tile(
            pl.TileType(shape=[16, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=256)
        pl.load(global_tile, features, [0, 0])
        pl.load(local_tile, features, [1, 0])
        pl.system.bar_all()
        pl.store(global_features, global_tile, [0, 0])
        pl.store(local_features, local_tile, [0, 0])
        pl.system.bar_all()


def _copy_coefficients(packed, destination, tile, start, rows, columns, source_col, stride, dest_col):
    pl.set_validshape(tile, [1, columns])
    for row in pl.range(rows):
        pl.load(tile, packed, [start + row, source_col])
        pl.system.bar_all()
        pointer = pl.addptr(pl.make_ptr(destination), row * stride + dest_col)
        # This one-row transfer writes only the valid columns at the final offset.
        output = pl.make_tensor(pointer, [1, columns], [16, 1])
        pl.store(output, tile, [0, 0])
        pl.system.bar_all()


@pl.jit(backend='cpu', arch='5102', auto_mutex=False, name='isp_unpack_coefficients_5102')
def unpack_coefficients(
    packed: pl.Tensor[[68, 16], pl.DT_FP16],
    channel_scale: pl.Tensor[[1, 4, 1, 1], pl.DT_FP16, pl.Output],
    global_exponent: pl.Tensor[[1, 1, 1, 1], pl.DT_FP16, pl.Output],
    global_bias: pl.Tensor[[1, 2, 1, 1], pl.DT_FP16, pl.Output],
    channel_mix: pl.Tensor[[1, 4, 4, 1, 1], pl.DT_FP16, pl.Output],
    tile_scale: pl.Tensor[[16, 4, 1, 1], pl.DT_FP16, pl.Output],
    tile_exponent: pl.Tensor[[16, 1, 1, 1], pl.DT_FP16, pl.Output],
    tile_bias: pl.Tensor[[16, 2, 1, 1], pl.DT_FP16, pl.Output],
    tile_mix: pl.Tensor[[16, 4, 4, 1, 1], pl.DT_FP16, pl.Output],
):
    with pl.section_vector():
        tile = pl.make_tile(
            pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                        valid_shape=[-1, -1]), addr=0)
        _copy_coefficients(packed, channel_scale, tile, 0, 1, 4, 0, 4, 0)
        _copy_coefficients(packed, global_exponent, tile, 1, 1, 1, 0, 1, 0)
        # GEMM emits (center, contrast). DecEnc reads (contrast, center).
        _copy_coefficients(packed, global_bias, tile, 2, 1, 1, 1, 2, 0)
        _copy_coefficients(packed, global_bias, tile, 2, 1, 1, 0, 2, 1)
        _copy_coefficients(packed, channel_mix, tile, 3, 1, 16, 0, 16, 0)
        _copy_coefficients(packed, tile_scale, tile, 4, 16, 4, 0, 4, 0)
        _copy_coefficients(packed, tile_exponent, tile, 20, 16, 1, 0, 1, 0)
        _copy_coefficients(packed, tile_bias, tile, 36, 16, 1, 1, 2, 0)
        _copy_coefficients(packed, tile_bias, tile, 36, 16, 1, 0, 2, 1)
        _copy_coefficients(packed, tile_mix, tile, 52, 16, 16, 0, 16, 0)
