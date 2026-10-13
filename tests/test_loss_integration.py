import pytest
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


@pytest.fixture
def trainer_data(tmp_path):
    """Keep trainer integration checks independent of downloads and COCO."""
    from paddy_yolo.dataset import generate_dummy_dataset, R2000_CLASSES

    root = generate_dummy_dataset(tmp_path / "data", num_images=2, img_size=(64, 64))
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({
        "path": root.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "names": R2000_CLASSES,
    }), encoding="utf-8")
    return str(config)


def test_imports_and_exports():
    """Verify PaddyBboxLoss, PaddyDetectionLoss, and PaddyTrainer can be imported from package roots."""
    from paddy_yolo.loss import PaddyBboxLoss, PaddyDetectionLoss
    from paddy_yolo.loss.paddy_loss import (
        PaddyBboxLoss as DirectPaddyBboxLoss,
        PaddyDetectionLoss as DirectPaddyDetectionLoss,
    )
    from paddy_yolo.engine import PaddyTrainer
    from paddy_yolo.engine.trainer import PaddyTrainer as DirectPaddyTrainer

    assert PaddyBboxLoss is DirectPaddyBboxLoss
    assert PaddyDetectionLoss is DirectPaddyDetectionLoss
    assert PaddyTrainer is DirectPaddyTrainer


def test_paddy_bbox_loss_computation():
    """Verify PaddyBboxLoss computes loss using DIMIoU Strategy H and backpropagates gradients."""
    from paddy_yolo.loss.paddy_loss import PaddyBboxLoss

    reg_max = 16
    bbox_loss_fn = PaddyBboxLoss(reg_max=reg_max, strategy="H")
    assert bbox_loss_fn.strategy == "H"

    num_fg = 4
    num_anchors = 10
    batch_size = 2

    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    fg_mask[0, :2] = True
    fg_mask[1, :2] = True

    pred_bboxes = (torch.rand(batch_size, num_anchors, 4) * 100).requires_grad_(True)
    pred_dist = (torch.rand(batch_size, num_anchors, 4 * reg_max) * 10).requires_grad_(True)
    target_bboxes = torch.rand(batch_size, num_anchors, 4) * 100
    anchor_points = torch.rand(num_anchors, 2) * 100
    target_scores = torch.rand(batch_size, num_anchors, 16)
    target_scores_sum = max(target_scores.sum(), 1.0)

    loss_iou, loss_dfl = bbox_loss_fn(
        pred_dist=pred_dist,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        target_bboxes=target_bboxes,
        target_scores=target_scores,
        target_scores_sum=target_scores_sum,
        fg_mask=fg_mask,
    )

    assert loss_iou.numel() == 1
    assert loss_dfl.numel() == 1
    assert loss_iou.item() >= 0.0
    assert loss_dfl.item() >= 0.0

    total_loss = loss_iou + loss_dfl
    total_loss.backward()

    assert pred_bboxes.grad is not None
    assert pred_dist.grad is not None
    assert not torch.isnan(pred_bboxes.grad).any()
    assert not torch.isnan(pred_dist.grad).any()


def test_paddy_detection_loss_forward_and_backward():
    """Verify PaddyDetectionLoss forward pass produces valid total_loss and loss_items tuple."""
    from paddy_yolo.modules.register import register_paddy_modules
    from paddy_yolo.loss.paddy_loss import PaddyBboxLoss, PaddyDetectionLoss

    register_paddy_modules()

    model = YOLO("configs/paddy-yolo.yaml").model
    loss_fn = PaddyDetectionLoss(model)

    assert isinstance(loss_fn.bbox_loss, PaddyBboxLoss)
    assert loss_fn.bbox_loss.strategy == "H"

    batch_size = 2
    img = torch.rand(batch_size, 3, 64, 64)
    preds = model(img)

    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0, 1]),
        "bboxes": torch.tensor([
            [0.1, 0.1, 0.4, 0.4],
            [0.2, 0.2, 0.6, 0.6],
        ]),
    }

    # Test callable loss_fn(preds, batch)
    total_loss, loss_items = loss_fn(preds, batch)

    assert total_loss.ndim == 0
    assert total_loss.requires_grad
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    # Contract: 3 loss items (box, cls, dfl), detached
    assert loss_items.shape == (3,)
    assert not loss_items.requires_grad
    assert torch.isfinite(loss_items).all()

    # Verify backward pass executes without error
    total_loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None and not torch.all(p.grad == 0):
            has_grad = True
            break
    assert has_grad, "Expected non-zero gradients on model parameters after loss backward"

    # Also test loss_fn.forward(preds, batch) alias
    total_loss_2, loss_items_2 = loss_fn.forward(preds, batch)
    assert total_loss_2.ndim == 0
    assert loss_items_2.shape == (3,)


def test_paddy_trainer_initialization_and_get_model(trainer_data, tmp_path):
    """Verify PaddyTrainer instantiates, registers modules, and configures model with PaddyDetectionLoss."""
    from paddy_yolo.loss.paddy_loss import PaddyDetectionLoss
    from paddy_yolo.engine.trainer import PaddyTrainer

    # Use small dataset/config overrides
    overrides = {
        "data": trainer_data,
        "project": str(tmp_path / "runs"),
        "model": "configs/paddy-yolo.yaml",
        "epochs": 1,
        "imgsz": 64,
        "batch": 2,
        "save": False,
        "plots": False,
    }
    trainer = PaddyTrainer(overrides=overrides)

    assert trainer is not None
    assert trainer.strategy == "H"

    # Verify get_model constructs model and binds init_criterion
    model = trainer.get_model(cfg="configs/paddy-yolo.yaml")
    assert isinstance(model, DetectionModel)
    assert hasattr(model, "init_criterion")

    criterion = model.init_criterion()
    assert isinstance(criterion, PaddyDetectionLoss)
    assert criterion.bbox_loss.strategy == "H"

    # Verify model.loss uses PaddyDetectionLoss
    batch_size = 2
    img = torch.rand(batch_size, 3, 64, 64)
    preds = model(img)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0, 1]),
        "bboxes": torch.tensor([
            [0.1, 0.1, 0.4, 0.4],
            [0.2, 0.2, 0.6, 0.6],
        ]),
    }
    loss_tuple = model.loss(batch, preds)
    assert isinstance(loss_tuple, tuple)
    assert len(loss_tuple) == 2
    assert isinstance(model.criterion, PaddyDetectionLoss)


def test_paddy_trainer_validation_loss_integration(trainer_data, tmp_path):
    """Verify PaddyTrainer callbacks/methods ensure PaddyDetectionLoss is used during validation."""
    from paddy_yolo.loss.paddy_loss import PaddyDetectionLoss
    from paddy_yolo.engine.trainer import PaddyTrainer
    from ultralytics.utils.torch_utils import ModelEMA

    overrides = {
        "data": trainer_data,
        "project": str(tmp_path / "runs"),
        "model": "configs/paddy-yolo.yaml",
        "epochs": 1,
        "imgsz": 64,
        "batch": 2,
        "save": False,
        "plots": False,
    }
    trainer = PaddyTrainer(overrides=overrides)
    trainer.model = trainer.get_model(cfg="configs/paddy-yolo.yaml")
    trainer.ema = ModelEMA(trainer.model)

    # Trigger val start callback
    mock_validator = type("MockValidator", (), {"model": None, "args": trainer.args})()
    for cb in trainer.callbacks.get("on_val_start", []):
        cb(mock_validator)

    # Check both model and ema.ema have PaddyDetectionLoss criterion configured
    assert isinstance(trainer.model.criterion, PaddyDetectionLoss)
    assert isinstance(trainer.ema.ema.criterion, PaddyDetectionLoss)
    assert trainer.model.criterion.bbox_loss.strategy == "H"
    assert trainer.ema.ema.criterion.bbox_loss.strategy == "H"


def test_paddy_validation_loss_unwraps_parallel_model():
    """A wrapped model must use the raw Detect module when rebuilding its loss."""
    from paddy_yolo.engine.trainer import PaddyTrainer
    from paddy_yolo.loss.paddy_loss import PaddyDetectionLoss

    trainer = object.__new__(PaddyTrainer)
    raw_model = YOLO("configs/paddy-yolo-ema-dysample.yaml").model
    trainer.model = torch.nn.DataParallel(raw_model)
    trainer._on_val_start()
    assert isinstance(raw_model.criterion, PaddyDetectionLoss)
    assert raw_model.init_criterion.model is raw_model
    assert raw_model.criterion.bbox_loss.strategy == "H"
