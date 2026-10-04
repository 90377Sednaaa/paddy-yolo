import pytest
import torch
import torch.nn as nn
from paddy_yolo.modules.ghostconv import GhostConv


def test_ghostconv_default_init():
    """Verify default initialization parameters match Paddy-YOLO specification."""
    c1, c2 = 16, 32
    layer = GhostConv(c1, c2)

    assert layer.k == 3
    assert layer.s == 1
    assert layer.dw_k == 5
    assert layer.c1 == c1
    assert layer.c2 == c2

    # Check primary convolution properties
    assert layer.cv1.conv.in_channels == c1
    assert layer.cv1.conv.out_channels == c2 // 2
    assert layer.cv1.conv.kernel_size == (3, 3)
    assert layer.cv1.conv.stride == (1, 1)
    assert layer.cv1.conv.padding == (1, 1)
    assert layer.cv1.conv.groups == 1

    # Check depthwise convolution properties
    assert layer.cv2.conv.in_channels == c2 // 2
    assert layer.cv2.conv.out_channels == c2 - c2 // 2
    assert layer.cv2.conv.kernel_size == (5, 5)
    assert layer.cv2.conv.stride == (1, 1)
    assert layer.cv2.conv.padding == (2, 2)
    assert layer.cv2.conv.groups == c2 // 2


def test_ghostconv_forward_shape():
    """Verify output shapes with stride=1."""
    x = torch.randn(2, 16, 64, 64)
    layer = GhostConv(16, 32)
    out = layer(x)
    assert out.shape == (2, 32, 64, 64)

    x2 = torch.randn(4, 32, 28, 28)
    layer2 = GhostConv(32, 64)
    out2 = layer2(x2)
    assert out2.shape == (4, 64, 28, 28)


def test_ghostconv_stride2_downsampling():
    """Verify output shapes with stride=2 (spatial downsampling)."""
    x = torch.randn(2, 16, 64, 64)
    layer = GhostConv(16, 32, s=2)
    out = layer(x)
    assert out.shape == (2, 32, 32, 32)


def test_ghostconv_custom_kernels():
    """Verify custom primary and depthwise kernel sizes."""
    layer = GhostConv(16, 32, k=1, dw_k=3)
    assert layer.cv1.conv.kernel_size == (1, 1)
    assert layer.cv2.conv.kernel_size == (3, 3)
    assert layer.cv2.conv.padding == (1, 1)

    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 32, 32, 32)


def test_ghostconv_gradient_flow():
    """Verify gradients propagate through both primary conv and depthwise conv."""
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    layer = GhostConv(16, 32)
    layer.train()

    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Input gradient
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Primary branch gradients
    assert layer.cv1.conv.weight.grad is not None
    assert layer.cv1.bn.weight.grad is not None
    assert layer.cv1.bn.bias.grad is not None

    # Cheap DWConv branch gradients
    assert layer.cv2.conv.weight.grad is not None
    assert layer.cv2.bn.weight.grad is not None
    assert layer.cv2.bn.bias.grad is not None


def test_ghostconv_channel_concatenation():
    """Verify exact channel concatenation: out = [y1, y2]."""
    x = torch.randn(2, 16, 32, 32)
    layer = GhostConv(16, 32)
    layer.eval()

    with torch.no_grad():
        out = layer(x)
        y1 = layer.cv1(x)
        y2 = layer.cv2(y1)
        expected = torch.cat([y1, y2], dim=1)

    assert torch.allclose(out, expected, atol=1e-6)
    assert out.shape[1] == 32
    assert y1.shape[1] == 16
    assert y2.shape[1] == 16


def test_ghostconv_activations():
    """Verify custom activations work properly."""
    x = torch.randn(2, 16, 32, 32)

    # act=False -> Identity
    layer_no_act = GhostConv(16, 32, act=False)
    assert isinstance(layer_no_act.cv1.act, nn.Identity)
    assert isinstance(layer_no_act.cv2.act, nn.Identity)
    out_no_act = layer_no_act(x)
    assert out_no_act.shape == (2, 32, 32, 32)

    # act=nn.ReLU() -> ReLU
    layer_relu = GhostConv(16, 32, act=nn.ReLU())
    assert isinstance(layer_relu.cv1.act, nn.ReLU)
    assert isinstance(layer_relu.cv2.act, nn.ReLU)
    out_relu = layer_relu(x)
    assert out_relu.shape == (2, 32, 32, 32)
    assert (out_relu >= 0.0).all()


def test_ghostconv_eval_and_train_modes():
    """Verify consistent behaviour across eval and train modes."""
    x = torch.randn(2, 16, 32, 32)
    layer = GhostConv(16, 32)

    layer.train()
    out_train = layer(x)
    assert out_train.shape == (2, 32, 32, 32)

    layer.eval()
    out_eval = layer(x)
    assert out_eval.shape == (2, 32, 32, 32)


def test_ghostconv_odd_channels():
    """Verify behavior when c2 is odd, ensuring output channels == c2."""
    x = torch.randn(2, 8, 16, 16)
    layer = GhostConv(8, 3)
    out = layer(x)
    assert out.shape == (2, 3, 16, 16)
    assert layer.cv1.conv.out_channels == 1
    assert layer.cv2.conv.out_channels == 2


def test_ghostconv_export_in_modules_init():
    """Verify GhostConv is exported from paddy_yolo.modules namespace."""
    from paddy_yolo.modules import GhostConv as ExportedGhostConv
    assert ExportedGhostConv is GhostConv
