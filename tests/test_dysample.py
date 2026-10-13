"""Unit tests for DySample dynamic upsampling module."""

import pytest
import torch
from paddy_yolo.modules.dysample import DySample


def test_dysample_shape_and_scale():
    """Verify DySample 2x spatial upsampling preserves channels and doubles H, W."""
    b, c, h, w = 2, 64, 20, 20
    x = torch.randn(b, c, h, w)
    dysample = DySample(in_channels=c, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (b, c, 40, 40), f"Expected (2, 64, 40, 40), got {out.shape}"


def test_dysample_gradient_flow():
    """Verify backpropagation through grid_sample and offset convolution."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    dysample = DySample(in_channels=32, scale=2, style="lp", groups=4)
    out = dysample(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input tensor gradient must not be None"
    assert x.grad.norm().item() > 0.0, "Input tensor gradient must be non-zero"
    assert dysample.offset.weight.grad is not None, "Offset conv gradient must not be None"


def test_dysample_channel_group_fallback():
    """Verify DySample gracefully handles cases where in_channels is smaller than groups."""
    x = torch.randn(1, 2, 10, 10)
    dysample = DySample(in_channels=2, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (1, 2, 20, 20)


def test_dysample_channels_last_memory_format():
    """Verify DySample works with channels_last memory format (crucial for GPU training)."""
    b, c, h, w = 4, 64, 20, 20
    x = torch.randn(b, c, h, w).contiguous(memory_format=torch.channels_last)
    assert not x.is_contiguous(), "Tensor must not be contiguous in standard layout"
    dysample = DySample(in_channels=c, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (b, c, 40, 40)


def test_dysample_non_contiguous_slice():
    """Verify DySample functions on strided non-contiguous sliced tensors."""
    x = torch.randn(2, 64, 40, 40)[:, :, ::2, ::2]
    assert not x.is_contiguous()
    dysample = DySample(in_channels=64, scale=2, style="lp", groups=4)
    out = dysample(x)
    assert out.shape == (2, 64, 40, 40)


def test_dysample_half_precision_amp():
    """Verify DySample behaves properly in FP16 / AMP half precision."""
    x = torch.randn(2, 64, 20, 20, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last
    )
    dysample = DySample(in_channels=64, scale=2, style="lp", groups=4).half()
    out = dysample(x)
    assert out.shape == (2, 64, 40, 40)
    assert out.dtype == torch.float16


def test_dysample_style_pl():
    """Verify DySample point-wise convolution (pl) style."""
    x = torch.randn(2, 64, 16, 16)
    dysample = DySample(in_channels=64, scale=2, style="pl", groups=4)
    out = dysample(x)
    assert out.shape == (2, 64, 32, 32)


def test_dysample_dyscope():
    """Verify DySample with dynamic scope adjustment enabled."""
    x = torch.randn(2, 64, 16, 16)
    dysample = DySample(in_channels=64, scale=2, style="lp", groups=4, dyscope=True)
    out = dysample(x)
    assert out.shape == (2, 64, 32, 32)


