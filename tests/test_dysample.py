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
