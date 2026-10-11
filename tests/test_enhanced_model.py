import torch
from pathlib import Path
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.modules.ema import EMA
from paddy_yolo.modules.rgg import RGG


def test_enhanced_paddy_yolo_yaml_exists():
    cfg = Path("configs/paddy-yolo-p2-ema.yaml")
    assert cfg.is_file(), f"Configuration file {cfg} not found"


def test_enhanced_paddy_yolo_build():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-p2-ema.yaml").model
    
    assert getattr(model, "nc", 16) == 16
    
    # Check 4 RGG modules in backbone
    rgg_layers = [m for m in model.model if isinstance(m, RGG)]
    assert len(rgg_layers) == 4, f"Expected 4 RGG backbone modules, found {len(rgg_layers)}"
    
    # Check 4 EMA modules across pyramid
    ema_layers = [m for m in model.model if isinstance(m, EMA)]
    assert len(ema_layers) == 4, f"Expected 4 EMA attention modules, found {len(ema_layers)}"


def test_enhanced_paddy_yolo_parameter_count():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-p2-ema.yaml").model
    
    train_params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < train_params < 3_600_000, f"Expected 3.0M-3.6M params, got {train_params}"
    
    # Test deploy mode
    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
            
    deploy_params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < deploy_params < 3_500_000, f"Expected 3.0M-3.5M deploy params, got {deploy_params}"


def test_enhanced_paddy_yolo_forward_anchors():
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-p2-ema.yaml").model
    model.eval()
    
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model(x)
        
    assert isinstance(preds, tuple)
    pred_boxes = preds[0]
    # Prediction shape: (batch, 4 + nc, total_anchors)
    # Total anchors with P2: 160*160 + 80*80 + 40*40 + 20*20 = 25600 + 6400 + 1600 + 400 = 34000
    assert pred_boxes.shape == (1, 20, 34000), f"Expected (1, 20, 34000), got {pred_boxes.shape}"
