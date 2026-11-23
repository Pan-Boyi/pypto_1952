# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
from pypto_1952.isp_1952_validation.reference_nv12 import to_yuv420_nv12


def test_nv12_plane_order():
    y=torch.tensor([[[[.1,.2]],[[.3,.4]],[[.5,.6]],[[.7,.8]]]],dtype=torch.float16)
    uv=torch.tensor([[[[.2,.4]],[[.6,.8]]]],dtype=torch.float16)
    expected=torch.tensor([[.1,.3,.2,.4,.5,.7,.6,.8,.2,.6,.4,.8]],dtype=torch.float16)
    assert torch.equal(to_yuv420_nv12(y,uv),expected)


def test_nv12_clipping_and_batches():
    y=torch.tensor([-1.,0.,1.,2.],dtype=torch.float16).view(1,4,1,1).repeat(2,1,1,1)
    uv=torch.tensor([-.5,1.5],dtype=torch.float16).view(1,2,1,1).repeat(2,1,1,1)
    expected=torch.tensor([[0.,0.,1.,1.,0.,1.]]*2,dtype=torch.float16)
    assert torch.equal(to_yuv420_nv12(y,uv),expected)
