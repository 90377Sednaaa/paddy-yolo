import pytest
import torch
import torch.nn as nn
from paddy_yolo.modules.repconv import RepConv


def test_repconv_forward():
    """Test standard forward pass in training mode."""
    x = torch.randn(2, 32, 64, 64)
    conv = RepConv(32, 32)
    conv.train()
    out_train = conv(x)
    assert out_train.shape == (2, 32, 64, 64)


def test_repconv_gradients():
    """Verify gradients propagate through all branches in training mode."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    conv = RepConv(32, 32)
    conv.train()
    out = conv(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert conv.conv1.conv.weight.grad is not None
    assert conv.conv2.conv.weight.grad is not None
    assert conv.bn.weight.grad is not None


def test_repconv_deploy_equivalence_c1_ne_c2():
    """Verify deploy equivalence when c1 != c2 (no identity branch)."""
    torch.manual_seed(42)
    x = torch.randn(1, 16, 32, 32)
    conv = RepConv(16, 32)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_after = conv(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)
    assert conv.deploy is True
    assert hasattr(conv, "conv")
    assert not hasattr(conv, "conv1")
    assert not hasattr(conv, "conv2")


def test_repconv_deploy_equivalence_c1_eq_c2():
    """Verify deploy equivalence when c1 == c2 and s == 1 (has identity BN branch)."""
    torch.manual_seed(42)
    x = torch.randn(2, 32, 32, 32)
    conv = RepConv(32, 32, s=1)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_after = conv(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)
    assert conv.deploy is True


def test_repconv_deploy_equivalence_stride2():
    """Verify deploy equivalence with stride=2 (spatial downsampling, no identity branch)."""
    torch.manual_seed(42)
    x = torch.randn(2, 32, 32, 32)
    conv = RepConv(32, 32, s=2)
    assert conv.bn is None  # no identity branch when s=2
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_after = conv(x)
    assert out_after.shape == (2, 32, 16, 16)
    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_repconv_deploy_equivalence_groups():
    """Verify deploy equivalence with groups > 1."""
    torch.manual_seed(42)
    x = torch.randn(2, 16, 16, 16)
    conv = RepConv(16, 16, g=2)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_after = conv(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_repconv_init_deploy_mode():
    """Verify direct initialization in deploy mode."""
    x = torch.randn(2, 16, 32, 32)
    conv = RepConv(16, 32, deploy=True)
    assert conv.deploy is True
    assert hasattr(conv, "conv")
    assert not hasattr(conv, "conv1")
    out = conv(x)
    assert out.shape == (2, 32, 32, 32)


def test_repconv_switch_to_deploy_idempotent():
    """Verify calling switch_to_deploy multiple times is safe and consistent."""
    torch.manual_seed(42)
    x = torch.randn(1, 16, 16, 16)
    conv = RepConv(16, 16)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.switch_to_deploy()
        out_first = conv(x)
        conv.switch_to_deploy()  # second call
        out_second = conv(x)
    assert torch.allclose(out_before, out_first, atol=1e-4)
    assert torch.allclose(out_first, out_second, atol=1e-6)


def test_repconv_activations():
    """Verify custom activation support (act=False and act=nn.ReLU())."""
    x = torch.randn(2, 16, 16, 16)

    # act=False -> Identity
    conv_no_act = RepConv(16, 16, act=False)
    assert isinstance(conv_no_act.act, nn.Identity)

    # act=nn.ReLU() -> ReLU
    conv_relu = RepConv(16, 16, act=nn.ReLU())
    assert isinstance(conv_relu.act, nn.ReLU)
    conv_relu.eval()
    with torch.no_grad():
        out_before = conv_relu(x)
        conv_relu.switch_to_deploy()
        out_after = conv_relu(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_repconv_ultralytics_compat():
    """Verify fuse_convs and forward_fuse aliases for Ultralytics compatibility."""
    x = torch.randn(1, 16, 16, 16)
    conv = RepConv(16, 16)
    conv.eval()
    with torch.no_grad():
        out_before = conv(x)
        conv.fuse_convs()
        out_fused = conv.forward_fuse(x)
    assert torch.allclose(out_before, out_fused, atol=1e-4)
    assert conv.rbr_reparam is conv.conv


def test_repconv_export_in_modules_init():
    """Verify RepConv is exported from paddy_yolo.modules namespace."""
    from paddy_yolo.modules import RepConv as ExportedRepConv
    assert ExportedRepConv is RepConv
