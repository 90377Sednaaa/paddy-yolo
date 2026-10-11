import torch
from pathlib import Path
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.modules.ema import EMA
from paddy_yolo.modules.rgg import RGG


def test_paddy_yolo_ema_config_exists():
    cfg = Path("configs/paddy-yolo-ema.yaml")
    assert cfg.is_file(), f"Configuration file {cfg} not found"


def test_paddy_yolo_ema_build():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-ema.yaml").model

    assert getattr(model, "nc", 16) == 16

    # Verify 4 RGG modules in backbone
    rgg_layers = [m for m in model.model if isinstance(m, RGG)]
    assert len(rgg_layers) == 4, f"Expected 4 RGG backbone modules, found {len(rgg_layers)}"

    # Verify 3 EMA modules before 3 detection heads
    ema_layers = [m for m in model.model if isinstance(m, EMA)]
    assert len(ema_layers) == 3, f"Expected 3 EMA attention modules, found {len(ema_layers)}"


def test_paddy_yolo_ema_parameter_count():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-ema.yaml").model

    train_params = sum(p.numel() for p in model.parameters())
    assert 3_300_000 < train_params < 3_450_000, f"Expected 3.3M-3.45M params, got {train_params}"

    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()

    deploy_params = sum(p.numel() for p in model.parameters())
    assert 3_300_000 < deploy_params < 3_400_000, f"Expected 3.3M-3.4M deploy params, got {deploy_params}"


def test_paddy_yolo_ema_forward_anchors():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-ema.yaml").model
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model(x)

    assert isinstance(preds, tuple)
    pred_boxes = preds[0]
    # Standard 3 heads: 80x80 + 40x40 + 20x20 = 6400 + 1600 + 400 = 8400
    assert pred_boxes.shape == (1, 20, 8400), f"Expected (1, 20, 8400), got {pred_boxes.shape}"
