"""Unchanged to_yuv420_nv12 math from the supplied module (line 1211)."""
import torch

def to_yuv420_nv12(y4, uv):
        B, C, H, W = y4.shape
        #y4, uv = torch.split(x, [4, 2], dim=1)
        # Y: pixel_shuffle(2) 等价展开, reshape 触发唯一一次 gather 拷贝
        y = y4.view(B, 2, 2, H, W).permute(0, 3, 1, 4, 2).reshape(B, 4 * H * W)
        # UV: (B,2,H,W) -> (B,H,W,2) -> reshape 得 [U0,V0,U1,V1,...] 交错
        uvi = uv.permute(0, 2, 3, 1).reshape(B, 2 * H * W)
        return torch.cat([y, uvi], dim=1).clamp(0,1)
