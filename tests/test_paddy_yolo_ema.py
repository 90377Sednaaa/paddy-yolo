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


def test_topological_weight_transfer_ema():
    """Verify that yolov8n.pt pretrained weights transfer topologically into backbone EMA."""
    weights_path = Path("yolov8n.pt")
    if not weights_path.exists():
        return  # Skip if weight file is missing in test environment

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-ema.yaml")
    init_sd = model.model.state_dict()

    src_ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    src_sd = src_ckpt["model"].state_dict() if isinstance(src_ckpt, dict) and "model" in src_ckpt else src_ckpt.state_dict()

    shift_maps = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10, 9: 12,
        12: 15, 15: 18, 16: 19, 18: 21, 19: 22, 21: 24, 22: 25,
    }

    transferred = 0
    for src_k, src_v in src_sd.items():
        if not src_k.startswith("model."):
            continue
        parts = src_k.split(".")
        src_idx = int(parts[1])
        if src_idx in shift_maps:
            tgt_idx = shift_maps[src_idx]
            tgt_k = f"model.{tgt_idx}." + ".".join(parts[2:])
            if tgt_k in init_sd and init_sd[tgt_k].shape == src_v.shape:
                transferred += 1

    assert transferred >= 280, f"Expected at least 280 transferred tensors, got {transferred}"

