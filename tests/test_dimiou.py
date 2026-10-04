import pytest
import torch
from paddy_yolo.loss.dimiou import dimiou, dimiou_loss, get_dynamic_ratio
from paddy_yolo.loss import (
    dimiou as exported_dimiou,
    dimiou_loss as exported_dimiou_loss,
    get_dynamic_ratio as exported_get_dynamic_ratio,
)


def test_package_exports():
    """Verify loss package exports dimiou, dimiou_loss, and get_dynamic_ratio."""
    assert exported_dimiou is dimiou
    assert exported_dimiou_loss is dimiou_loss
    assert exported_get_dynamic_ratio is get_dynamic_ratio


def test_dimiou_identical_boxes():
    """Verify identical boxes yield loss=0.0 and metric=1.0."""
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

    metric = dimiou(b1, b2)
    loss = dimiou_loss(b1, b2)

    assert torch.isclose(loss, torch.tensor([[0.0]]), atol=1e-3)
    assert torch.isclose(metric, torch.tensor([[1.0]]), atol=1e-3)


def test_dimiou_shifted_boxes():
    """Verify partially overlapping boxes have positive loss and metric in (0, 1)."""
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2 = torch.tensor([[20.0, 20.0, 60.0, 60.0]])

    metric = dimiou(b1, b2)
    loss = dimiou_loss(b1, b2)

    assert loss.item() > 0.0
    assert 0.0 < metric.item() < 1.0
    assert torch.isclose(loss + metric, torch.tensor([[1.0]]), atol=1e-4)


def test_dimiou_disjoint_boxes():
    """Verify non-overlapping boxes have loss > 1.0 due to MPDIoU corner distance penalty."""
    b1 = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    b2 = torch.tensor([[100.0, 100.0, 120.0, 120.0]])

    loss = dimiou_loss(b1, b2)
    metric = dimiou(b1, b2)

    assert loss.item() > 1.0
    assert metric.item() < 0.0
    assert torch.isclose(loss + metric, torch.tensor([[1.0]]), atol=1e-4)


def test_dimiou_gradient_backprop():
    """Verify gradients propagate cleanly to predicted box coordinates."""
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]], requires_grad=True)
    b2 = torch.tensor([[15.0, 15.0, 55.0, 55.0]])

    loss = dimiou_loss(b1, b2)
    loss.sum().backward()

    assert b1.grad is not None
    assert not torch.isnan(b1.grad).any()
    assert not torch.isinf(b1.grad).any()
    assert not torch.all(b1.grad == 0)

    # Disjoint boxes should also backpropagate non-zero gradient via MPDIoU corner penalties
    b1_disjoint = torch.tensor([[10.0, 10.0, 20.0, 20.0]], requires_grad=True)
    b2_disjoint = torch.tensor([[100.0, 100.0, 120.0, 120.0]])

    loss_disjoint = dimiou_loss(b1_disjoint, b2_disjoint)
    loss_disjoint.sum().backward()

    assert b1_disjoint.grad is not None
    assert not torch.isnan(b1_disjoint.grad).any()
    assert not torch.all(b1_disjoint.grad == 0)


def test_dimiou_xywh_vs_xyxy():
    """Verify xywh=True matches xywh=False given equivalent bounding box definitions."""
    # xyxy: [x1, y1, x2, y2]
    xyxy_b1 = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    xyxy_b2 = torch.tensor([[15.0, 25.0, 55.0, 85.0]])

    # xywh: center x, center y, w, h -> [30, 50, 40, 60], [35, 55, 40, 60]
    xywh_b1 = torch.tensor([[30.0, 50.0, 40.0, 60.0]])
    xywh_b2 = torch.tensor([[35.0, 55.0, 40.0, 60.0]])

    loss_xyxy = dimiou_loss(xyxy_b1, xyxy_b2, xywh=False)
    loss_xywh = dimiou_loss(xywh_b1, xywh_b2, xywh=True)

    assert torch.allclose(loss_xyxy, loss_xywh, atol=1e-5)

    metric_xyxy = dimiou(xyxy_b1, xyxy_b2, xywh=False)
    metric_xywh = dimiou(xywh_b1, xywh_b2, xywh=True)

    assert torch.allclose(metric_xyxy, metric_xywh, atol=1e-5)


def test_strategy_h_ratio_curve():
    """Verify Strategy H ratio values match Eq. 19: ratio = 0.5 * cos((1 - iou) * pi) + 1.0."""
    # Key anchor points
    r_0 = get_dynamic_ratio(torch.tensor(0.0), strategy="H")
    r_half = get_dynamic_ratio(torch.tensor(0.5), strategy="H")
    r_1 = get_dynamic_ratio(torch.tensor(1.0), strategy="H")

    assert torch.isclose(r_0, torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(r_half, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(r_1, torch.tensor(1.5), atol=1e-5)

    # Monotonically increasing across [0, 1]
    ious = torch.linspace(0.0, 1.0, 50)
    ratios = get_dynamic_ratio(ious, strategy="H")
    diffs = ratios[1:] - ratios[:-1]
    assert (diffs >= -1e-6).all(), "Strategy H ratio must be monotonically increasing"


def test_all_dynamic_strategies_a_through_l():
    """Verify all strategies A through L execute and return valid ratio values in [0.5, 1.5]."""
    ious = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    strategies = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2 = torch.tensor([[15.0, 15.0, 55.0, 55.0]])

    for s in strategies:
        # Check ratio function
        ratio = get_dynamic_ratio(ious, strategy=s)
        assert (ratio >= 0.49).all() and (ratio <= 1.51).all(), f"Strategy {s} ratio out of bounds: {ratio}"

        # Case insensitivity
        ratio_lower = get_dynamic_ratio(ious, strategy=s.lower())
        assert torch.allclose(ratio, ratio_lower)

        # Check loss computation with this strategy
        loss = dimiou_loss(b1, b2, strategy=s)
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0.0

    # Invalid strategy raises ValueError
    with pytest.raises(ValueError, match="Unknown DIMIoU strategy"):
        get_dynamic_ratio(ious, strategy="Z")

    with pytest.raises(ValueError, match="Unknown DIMIoU strategy"):
        dimiou_loss(b1, b2, strategy="UNKNOWN")


def test_dimiou_batch_shapes():
    """Verify DIMIoU handles multi-dimensional batches and broadcasting."""
    # (N, 4)
    b1_n = torch.rand(10, 4) * 100
    b2_n = torch.rand(10, 4) * 100
    loss_n = dimiou_loss(b1_n, b2_n, xywh=True)
    assert loss_n.shape == (10, 1)

    # (B, N, 4)
    b1_bn = torch.rand(2, 5, 4) * 100
    b2_bn = torch.rand(2, 5, 4) * 100
    loss_bn = dimiou_loss(b1_bn, b2_bn, xywh=True)
    assert loss_bn.shape == (2, 5, 1)

    # Broadcasting (1, 4) against (N, 4)
    b1_single = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2_multi = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [20.0, 20.0, 60.0, 60.0],
        [100.0, 100.0, 120.0, 120.0],
    ])
    loss_broadcast = dimiou_loss(b1_single, b2_multi)
    assert loss_broadcast.shape == (3, 1)
    assert torch.isclose(loss_broadcast[0, 0], torch.tensor(0.0), atol=1e-3)
    assert loss_broadcast[1, 0].item() > 0.0
    assert loss_broadcast[2, 0].item() > 1.0
