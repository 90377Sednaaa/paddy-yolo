import pytest
import torch
import torch.nn as nn
from paddy_yolo.modules import RGG, RGGBottleneck, RGGConv


def test_rgg_export_in_modules_init():
    """Verify RGG, RGGBottleneck, and RGGConv are exported in paddy_yolo.modules."""
    import paddy_yolo.modules as pym
    assert hasattr(pym, "RGGConv")
    assert hasattr(pym, "RGGBottleneck")
    assert hasattr(pym, "RGG")
    assert "RGGConv" in pym.__all__
    assert "RGGBottleneck" in pym.__all__
    assert "RGG" in pym.__all__


def test_rggconv_initialization():
    """Verify RGGConv module components and channel configuration."""
    c1, c2 = 16, 32
    layer = RGGConv(c1, c2)

    assert hasattr(layer, "rep")
    assert hasattr(layer, "ghost1")
    assert hasattr(layer, "ghost2")
    assert hasattr(layer, "conv")

    # RepConv input/output channels
    assert layer.rep.c1 == c1
    assert layer.rep.c2 == c2

    # GhostConv input/output channels and kernel configuration
    assert layer.ghost1.c1 == c1
    assert layer.ghost1.c2 == c2
    assert layer.ghost1.k == 3
    assert layer.ghost1.dw_k == 5

    assert layer.ghost2.c1 == c1
    assert layer.ghost2.c2 == c2
    assert layer.ghost2.k == 3
    assert layer.ghost2.dw_k == 5

    # 1x1 projection conv takes 3 * c2 -> c2
    assert layer.conv.conv.in_channels == 3 * c2
    assert layer.conv.conv.out_channels == c2
    assert layer.conv.conv.kernel_size == (1, 1)


def test_rggconv_forward_shape():
    """Verify RGGConv output shapes with matching and differing channels."""
    # c1 == c2
    x1 = torch.randn(2, 32, 20, 20)
    layer1 = RGGConv(32, 32)
    out1 = layer1(x1)
    assert out1.shape == (2, 32, 20, 20)

    # c1 != c2
    x2 = torch.randn(4, 16, 40, 40)
    layer2 = RGGConv(16, 48)
    out2 = layer2(x2)
    assert out2.shape == (4, 48, 40, 40)


def test_rggconv_gradients():
    """Verify gradients propagate through all branches and projection conv in RGGConv."""
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    layer = RGGConv(16, 32)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any()

    # Verify all internal components have gradients
    assert layer.rep.conv1.conv.weight.grad is not None
    assert layer.rep.conv2.conv.weight.grad is not None
    assert layer.ghost1.cv1.conv.weight.grad is not None
    assert layer.ghost1.cv2.conv.weight.grad is not None
    assert layer.ghost2.cv1.conv.weight.grad is not None
    assert layer.ghost2.cv2.conv.weight.grad is not None
    assert layer.conv.conv.weight.grad is not None


def test_rggconv_deploy_equivalence():
    """Verify RGGConv switch_to_deploy produces identical output in eval mode."""
    x = torch.randn(2, 16, 16, 16)
    layer = RGGConv(16, 32)
    layer.eval()

    with torch.no_grad():
        out_train_mode = layer(x)

    layer.switch_to_deploy()
    assert layer.rep.deploy is True

    with torch.no_grad():
        out_deploy_mode = layer(x)

    torch.testing.assert_close(out_train_mode, out_deploy_mode, rtol=1e-4, atol=1e-4)


def test_rgg_bottleneck_initialization():
    """Verify RGGBottleneck initialization and shortcut condition."""
    # Matching channels with shortcut=True
    b1 = RGGBottleneck(32, 32, shortcut=True)
    assert b1.add is True
    assert b1.cv1.conv.in_channels == 32
    assert b1.cv1.conv.out_channels == 32
    assert b1.cv1.conv.kernel_size == (3, 3)

    # Differing channels with shortcut=True
    b2 = RGGBottleneck(16, 32, shortcut=True)
    assert b2.add is False

    # Matching channels with shortcut=False
    b3 = RGGBottleneck(32, 32, shortcut=False)
    assert b3.add is False


def test_rgg_bottleneck_forward_shape():
    """Verify RGGBottleneck forward shapes."""
    x = torch.randn(2, 32, 20, 20)
    block = RGGBottleneck(32, 32, shortcut=True)
    out = block(x)
    assert out.shape == (2, 32, 20, 20)

    x2 = torch.randn(2, 16, 24, 24)
    block2 = RGGBottleneck(16, 32, shortcut=False)
    out2 = block2(x2)
    assert out2.shape == (2, 32, 24, 24)


def test_rgg_bottleneck_residual_behavior():
    """Verify RGGBottleneck residual addition behavior."""
    x = torch.randn(2, 32, 16, 16)
    block_with_sc = RGGBottleneck(32, 32, shortcut=True)
    block_no_sc = RGGBottleneck(32, 32, shortcut=False)

    # Clone state dict so both blocks have identical weights
    block_no_sc.load_state_dict(block_with_sc.state_dict())

    block_with_sc.eval()
    block_no_sc.eval()

    with torch.no_grad():
        out_sc = block_with_sc(x)
        out_no_sc = block_no_sc(x)

    # The difference must exactly equal the residual connection x
    diff = out_sc - out_no_sc
    torch.testing.assert_close(diff, x, rtol=1e-5, atol=1e-5)


def test_rgg_bottleneck_gradients():
    """Verify gradients propagate through RGGBottleneck."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    block = RGGBottleneck(32, 32, shortcut=True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert (x.grad != 0).any()
    assert block.cv1.conv.weight.grad is not None
    assert block.cv2.conv.conv.weight.grad is not None


def test_rgg_module_initialization():
    """Verify RGG module channel scaling and bottleneck list creation."""
    c1, c2, n, e = 64, 128, 3, 0.5
    rgg = RGG(c1, c2, n=n, shortcut=True, e=e)

    expected_c = int(c2 * e)  # 64
    assert rgg.c == expected_c
    assert rgg.cv1.conv.in_channels == c1
    assert rgg.cv1.conv.out_channels == 2 * expected_c
    assert rgg.cv2.conv.in_channels == (2 + n) * expected_c
    assert rgg.cv2.conv.out_channels == c2
    assert len(rgg.m) == n

    for m in rgg.m:
        assert isinstance(m, RGGBottleneck)
        assert m.add is True


def test_rgg_module_forward_shape():
    """Verify RGG module forward shape across various configurations."""
    # Standard n=1
    x1 = torch.randn(2, 32, 40, 40)
    rgg1 = RGG(32, 32, n=1, shortcut=True)
    out1 = rgg1(x1)
    assert out1.shape == (2, 32, 40, 40)

    # Backbone Layer 2 style (c1=32, c2=64, n=2)
    x2 = torch.randn(2, 64, 40, 40)
    rgg2 = RGG(64, 64, n=2, shortcut=True)
    out2 = rgg2(x2)
    assert out2.shape == (2, 64, 40, 40)

    # Large backbone stage (c1=128, c2=256, n=3)
    x3 = torch.randn(1, 128, 20, 20)
    rgg3 = RGG(128, 256, n=3, shortcut=True)
    out3 = rgg3(x3)
    assert out3.shape == (1, 256, 20, 20)


def test_rgg_forward_split_equivalence():
    """Verify RGG forward and forward_split produce identical results."""
    x = torch.randn(2, 64, 32, 32)
    rgg = RGG(64, 64, n=2, shortcut=True)
    rgg.eval()

    with torch.no_grad():
        out_forward = rgg(x)
        out_split = rgg.forward_split(x)

    torch.testing.assert_close(out_forward, out_split)


def test_rgg_module_gradients():
    """Verify gradients propagate through all layers of RGG module."""
    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    rgg = RGG(64, 64, n=2, shortcut=True)
    out = rgg(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert (x.grad != 0).any()
    assert rgg.cv1.conv.weight.grad is not None
    assert rgg.cv2.conv.weight.grad is not None
    for m in rgg.m:
        assert m.cv1.conv.weight.grad is not None
        assert m.cv2.conv.conv.weight.grad is not None


def test_rgg_switch_to_deploy():
    """Verify RGG switch_to_deploy collapses all RepConvs and matches eval output."""
    x = torch.randn(2, 64, 20, 20)
    rgg = RGG(64, 64, n=2, shortcut=True)
    rgg.eval()

    with torch.no_grad():
        out_train_mode = rgg(x)

    rgg.switch_to_deploy()
    for m in rgg.m:
        assert m.cv2.rep.deploy is True

    with torch.no_grad():
        out_deploy_mode = rgg(x)

    torch.testing.assert_close(out_train_mode, out_deploy_mode, rtol=1e-4, atol=1e-4)


def test_rgg_ultralytics_calling_convention():
    """Verify RGG supports positional arguments passed by Ultralytics parse_model."""
    # Ultralytics parse_model calls: m(c1, c2, *args) where args might be [n, shortcut]
    # e.g., RGG(32, 64, 2, True)
    layer = RGG(32, 64, 2, True)
    assert layer.c == 32
    assert len(layer.m) == 2
    assert layer.m[0].add is True

    # e.g., RGG(64, 128, 3)
    layer2 = RGG(64, 128, 3)
    assert layer2.c == 64
    assert len(layer2.m) == 3
    assert layer2.m[0].add is True
