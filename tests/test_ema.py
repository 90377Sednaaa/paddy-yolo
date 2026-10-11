import torch
import pytest
from paddy_yolo.modules.ema import EMA


def test_ema_initialization():
    ema = EMA(channels=32, factor=8)
    assert ema.groups == 8
    assert ema.conv1x1.in_channels == 4
    assert ema.conv3x3.in_channels == 4


def test_ema_forward_shape():
    for ch in [32, 64, 128, 256]:
        ema = EMA(channels=ch, factor=8)
        x = torch.randn(2, ch, 32, 32)
        out = ema(x)
        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


def test_ema_gradient_flow():
    ema = EMA(channels=64, factor=8)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    out = ema(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any()


def test_ema_non_divisible_channels_fallback():
    # If channels is not divisible by 8, it should automatically choose a valid factor
    ema = EMA(channels=12, factor=8)
    assert 12 % ema.groups == 0
    x = torch.randn(2, 12, 16, 16)
    out = ema(x)
    assert out.shape == x.shape


def test_ema_eval_and_train_modes():
    ema = EMA(channels=32, factor=8)
    x = torch.randn(2, 32, 16, 16)
    
    ema.train()
    out_train = ema(x)
    
    ema.eval()
    with torch.no_grad():
        out_eval = ema(x)
    
    assert out_train.shape == out_eval.shape


def test_ema_residual_connection():
    """Verify that EMA includes a residual identity skip connection preventing signal vanishing."""
    ema = EMA(channels=32, factor=8)
    x = torch.randn(2, 32, 16, 16)
    out = ema(x)
    # Output should differ from raw x but be positively correlated due to the residual identity x + att
    assert out.shape == x.shape
    # Check that output is not zero even if attention weights were tiny
    assert out.abs().sum() > 0
    # Check gradient flow through identity branch
    x_grad = torch.randn(2, 32, 16, 16, requires_grad=True)
    out_grad = ema(x_grad)
    out_grad.sum().backward()
    assert (x_grad.grad > 0.5).any(), "Expected gradient contribution through residual identity branch"

