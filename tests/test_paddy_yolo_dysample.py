"""Tests for Paddy-YOLO with DySample architecture loading and forward pass."""

import torch
from paddy_yolo.modules.register import register_paddy_modules
from ultralytics import YOLO


def test_paddy_yolo_dysample_build():
    """Verify configs/paddy-yolo-dysample.yaml builds and runs forward pass."""
    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-dysample.yaml")

    dummy_input = torch.randn(1, 3, 640, 640)
    preds = model(dummy_input, verbose=False)

    assert preds is not None
    # Check total parameter count is approximately 3.16M params
    param_count = sum(p.numel() for p in model.model.parameters())
    assert 3_000_000 < param_count < 3_500_000, f"Unexpected param count: {param_count}"


def test_pretrained_weights_transfer():
    """Verify transferring weights from yolov8n.pt into Paddy-YOLO-DySample."""
    from pathlib import Path

    weights_path = Path("yolov8n.pt")
    if not weights_path.is_file():
        return

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo-dysample.yaml")
    model.load(str(weights_path))

    # Verify model forward pass succeeds after loading weights
    out = model(torch.randn(1, 3, 640, 640), verbose=False)
    assert out is not None
