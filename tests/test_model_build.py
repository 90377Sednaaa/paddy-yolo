import os
import pytest
import torch
from ultralytics import YOLO


def test_registration_modules():
    """Verify register_paddy_modules registers custom modules into ultralytics namespaces."""
    from paddy_yolo.modules.register import register_paddy_modules
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks

    register_paddy_modules()

    for name in ["RGG", "RGGBottleneck", "RGGConv", "RepConv", "GhostConv"]:
        assert hasattr(modules, name), f"{name} missing from ultralytics.nn.modules"
        assert hasattr(block, name), f"{name} missing from ultralytics.nn.modules.block"
        assert hasattr(tasks, name), f"{name} missing from ultralytics.nn.tasks"


def test_auto_registration_on_import():
    """Verify importing paddy_yolo automatically triggers registration."""
    import paddy_yolo
    import ultralytics.nn.tasks as tasks

    assert hasattr(paddy_yolo, "register_paddy_modules")
    assert hasattr(tasks, "RGG")


def test_paddy_yolo_model_build():
    """Verify YOLO('configs/paddy-yolo.yaml') builds and runs a training forward pass."""
    from paddy_yolo.modules.register import register_paddy_modules
    from paddy_yolo.modules import RGG

    register_paddy_modules()
    yaml_path = "configs/paddy-yolo.yaml"
    assert os.path.exists(yaml_path), f"Architecture config {yaml_path} not found"

    model = YOLO(yaml_path)
    assert model is not None
    assert model.model is not None

    x = torch.randn(1, 3, 640, 640)
    out = model.model(x)
    assert out is not None


def test_paddy_yolo_backbone_layers():
    """Verify backbone layers 2, 4, 6, 8 are RGG and head layers are C2f."""
    from paddy_yolo.modules.register import register_paddy_modules
    from paddy_yolo.modules import RGG
    from ultralytics.nn.modules import C2f

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo.yaml")
    layers = list(model.model.model)

    # Backbone RGG layers
    assert isinstance(layers[2], RGG), f"Layer 2 is {type(layers[2])}, expected RGG"
    assert isinstance(layers[4], RGG), f"Layer 4 is {type(layers[4])}, expected RGG"
    assert isinstance(layers[6], RGG), f"Layer 6 is {type(layers[6])}, expected RGG"
    assert isinstance(layers[8], RGG), f"Layer 8 is {type(layers[8])}, expected RGG"

    # Head C2f layers
    assert isinstance(layers[12], C2f), f"Layer 12 is {type(layers[12])}, expected C2f"
    assert isinstance(layers[15], C2f), f"Layer 15 is {type(layers[15])}, expected C2f"
    assert isinstance(layers[18], C2f), f"Layer 18 is {type(layers[18])}, expected C2f"
    assert isinstance(layers[21], C2f), f"Layer 21 is {type(layers[21])}, expected C2f"


def test_paddy_yolo_params_and_classes():
    """Verify model parameter count and number of classes match paper specifications."""
    from paddy_yolo.modules.register import register_paddy_modules

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo.yaml")

    # 16 rice pest classes
    assert model.model.yaml["nc"] == 16
    assert model.model.model[-1].nc == 16
    assert len(model.model.names) == 16

    # Parameters expectation: ~3.16M - 3.3M params
    total_params = sum(p.numel() for p in model.model.parameters())
    assert 3_000_000 <= total_params <= 3_500_000, f"Unexpected parameter count: {total_params}"


def test_paddy_yolo_eval_forward_shape():
    """Verify eval mode forward pass produces expected shape (1, 20, 8400) for 16 classes."""
    from paddy_yolo.modules.register import register_paddy_modules

    register_paddy_modules()
    model = YOLO("configs/paddy-yolo.yaml")
    model.model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model.model(x)

    # In eval mode, Detect head returns (preds, train_outputs) or tensor
    pred_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
    # Shape: (batch=1, channels=4+16=20, anchors=8400)
    assert pred_tensor.shape == (1, 20, 8400), f"Expected shape (1, 20, 8400), got {pred_tensor.shape}"
