"""Unit and integration tests for Paddy-YOLO with EMA and DySample (Configuration D).

Validates:
1. Dynamic registration of custom modules (RGG, EMA, DySample).
2. Architecture parsing from configs/paddy-yolo-ema-dysample.yaml.
3. Module placement: EMA in backbone (layers 5, 8, 11) and DySample in neck (layers 13, 16).
4. Forward and backward pass gradient flow.
5. COCO pre-trained weight transfer compatibility via shift mapping.
"""

import pytest
import torch
from pathlib import Path
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.modules.ema import EMA
from paddy_yolo.modules.dysample import DySample
from paddy_yolo.modules.rgg import RGG


@pytest.fixture(scope="module", autouse=True)
def setup_modules():
    """Ensure custom Paddy-YOLO modules are registered into Ultralytics."""
    register_paddy_modules()


def test_paddy_yolo_ema_dysample_architecture():
    """Verify Paddy-YOLO-EMA-DySample parses and contains EMA and DySample at expected layers."""
    cfg_path = Path("configs/paddy-yolo-ema-dysample.yaml")
    assert cfg_path.is_file(), f"Configuration file not found: {cfg_path}"

    yolo = YOLO(str(cfg_path))
    model = yolo.model
    assert model is not None

    layers = list(model.model.children())
    assert len(layers) == 26, f"Expected 26 layers (0-25), found {len(layers)}"

    # Check EMA layers in backbone
    assert isinstance(layers[5], EMA), f"Layer 5 should be EMA, got {type(layers[5])}"
    assert isinstance(layers[8], EMA), f"Layer 8 should be EMA, got {type(layers[8])}"
    assert isinstance(layers[11], EMA), f"Layer 11 should be EMA, got {type(layers[11])}"

    # Check DySample layers in neck
    assert isinstance(layers[13], DySample), f"Layer 13 should be DySample, got {type(layers[13])}"
    assert isinstance(layers[16], DySample), f"Layer 16 should be DySample, got {type(layers[16])}"

    # Check Detect head
    from ultralytics.nn.modules.head import Detect
    assert isinstance(layers[25], Detect), f"Layer 25 should be Detect, got {type(layers[25])}"
    assert layers[25].nc == 16, f"Expected 16 classes, got {layers[25].nc}"


def test_paddy_yolo_ema_dysample_forward_pass():
    """Verify forward inference and output tensor shapes."""
    yolo = YOLO("configs/paddy-yolo-ema-dysample.yaml")
    model = yolo.model
    model.eval()

    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model(dummy_input)

    # In eval mode with Detect head, output is typically tuple (inference_tensor, loss_inputs)
    if isinstance(preds, (tuple, list)):
        out = preds[0]
    else:
        out = preds

    # Shape should be [1, 4 + 16, 8400] for 640x640 with nc=16
    assert out.shape[0] == 1
    assert out.shape[1] == 20  # 4 bbox coords + 16 classes
    assert out.shape[2] == 8400  # (80*80 + 40*40 + 20*20) anchor points


def test_paddy_yolo_ema_dysample_backward_pass():
    """Verify gradient backpropagation through both EMA and DySample modules."""
    yolo = YOLO("configs/paddy-yolo-ema-dysample.yaml")
    model = yolo.model
    model.train()

    dummy_input = torch.randn(2, 3, 640, 640, requires_grad=True)
    preds = model(dummy_input)

    # In training mode, preds is a list of 3 feature maps from the Detect head
    loss = sum(p.sum() for p in preds)
    loss.backward()

    # Verify gradients reach input
    assert dummy_input.grad is not None
    assert not torch.isnan(dummy_input.grad).any()

    # Verify gradients reach EMA layers
    layers = list(model.model.children())
    for ema_idx in [5, 8, 11]:
        ema_layer = layers[ema_idx]
        assert ema_layer.conv1x1.weight.grad is not None
        assert not torch.isnan(ema_layer.conv1x1.weight.grad).any()

    # Verify gradients reach DySample layers
    for dysample_idx in [13, 16]:
        dysample_layer = layers[dysample_idx]
        assert dysample_layer.offset.weight.grad is not None
        assert not torch.isnan(dysample_layer.offset.weight.grad).any()


def test_paddy_yolo_ema_dysample_weight_transfer():
    """Verify topological COCO pre-trained weight transfer from yolov8n.pt."""
    weights_path = Path("yolov8n.pt")
    if not weights_path.exists():
        pytest.skip("yolov8n.pt not present locally for weight transfer test")

    yolo = YOLO("configs/paddy-yolo-ema-dysample.yaml")
    yolo.load(str(weights_path))

    # Apply shift mapping
    ema_shift_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10, 9: 12,
        12: 15, 15: 18, 16: 19, 18: 21, 19: 22, 21: 24, 22: 25,
    }
    src_ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    src_sd = src_ckpt["model"].state_dict() if isinstance(src_ckpt, dict) and "model" in src_ckpt else src_ckpt.state_dict()
    tgt_sd = yolo.model.state_dict()

    transferred = 0
    for src_key, src_val in src_sd.items():
        if not src_key.startswith("model."):
            continue
        parts = src_key.split(".")
        src_idx = int(parts[1])
        if src_idx in ema_shift_map:
            tgt_key = f"model.{ema_shift_map[src_idx]}." + ".".join(parts[2:])
            if tgt_key in tgt_sd and tgt_sd[tgt_key].shape == src_val.shape:
                tgt_sd[tgt_key].copy_(src_val.float())
                transferred += 1

    yolo.model.load_state_dict(tgt_sd)
    assert transferred > 40, f"Expected >40 transferred tensors, got {transferred}"
