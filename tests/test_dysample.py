"""DySample integration checks for the separate Paddy-YOLO neck variant."""

from pathlib import Path

import pytest
import torch
import yaml
from ultralytics import YOLO


def test_original_dysample_initialization_and_gradients():
    from paddy_yolo.modules.dysample import DySample

    layer = DySample(256, scale=2, style="lp", groups=4, dyscope=False)
    assert layer.offset.in_channels == 256
    assert layer.offset.out_channels == 32
    assert not hasattr(layer, "scope")
    assert layer.init_pos.shape == (1, 32, 1, 1)
    assert torch.all(layer.offset.bias == 0)
    x = torch.randn(2, 256, 8, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 256, 16, 16)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.offset.weight.grad is not None and layer.offset.weight.grad.abs().sum() > 0


def test_proposed_yaml_changes_only_the_two_neck_upsamples():
    baseline = yaml.safe_load(Path("configs/paddy-yolo.yaml").read_text())
    proposal = yaml.safe_load(Path("configs/Paddy-YOLO-DySample.yaml").read_text())
    assert proposal["backbone"] == baseline["backbone"]
    assert proposal["head"][-1] == baseline["head"][-1]
    assert proposal["nc"] == baseline["nc"] == 16
    for index in range(len(baseline["head"])):
        if index in (0, 3):
            assert baseline["head"][index][2] == "nn.Upsample"
            assert proposal["head"][index] == [-1, 1, "DySample", [2, "lp", 4, False]]
        else:
            assert proposal["head"][index] == baseline["head"][index]


def test_proposed_model_build_forward_backward_and_head_shapes():
    from paddy_yolo.modules.dysample import DySample
    from paddy_yolo.modules.rgg import RGG

    model = YOLO("configs/Paddy-YOLO-DySample.yaml").model
    layers = model.model
    assert [i for i, layer in enumerate(layers) if isinstance(layer, DySample)] == [10, 13]
    assert [i for i, layer in enumerate(layers) if isinstance(layer, RGG)] == [2, 4, 6, 8]
    assert layers[10].offset.in_channels == 256
    assert layers[13].offset.in_channels == 128
    assert layers[10].groups == layers[13].groups == 4

    seen = []
    handle = layers[22].register_forward_pre_hook(
        lambda _module, inputs: seen.append([tuple(feature.shape) for feature in inputs[0]])
    )
    try:
        model.train()
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        outputs = model(x)
        assert [shape[2:] for shape in seen[-1]] == [(8, 8), (4, 4), (2, 2)]
        assert [shape[1] for shape in seen[-1]] == [64, 128, 256]
        sum(output.square().mean() for output in outputs).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for index in (10, 13):
            grad = layers[index].offset.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    finally:
        handle.remove()


@pytest.mark.parametrize("config", ["configs/paddy-yolo.yaml", "configs/Paddy-YOLO-DySample.yaml"])
def test_both_model_configs_keep_three_detection_scales(config):
    model = YOLO(config).model.eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 64, 64))
    predictions = result[0] if isinstance(result, (tuple, list)) else result
    assert predictions.shape == (1, 20, 84)


def test_dysample_plus_preserves_features_and_trains_scope_branches():
    """Dynamic scope must be active at both neck stages and receive gradients."""
    from paddy_yolo.modules.dysample import DySample

    reference = YOLO("configs/Paddy-YOLO-DySample.yaml").model
    model = YOLO("configs/Paddy-YOLO-DySample-Plus.yaml").model
    assert model.yaml["backbone"] == reference.yaml["backbone"]
    assert model.yaml["head"][-1] == reference.yaml["head"][-1]
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in reference.parameters()) == 12288
    seen = []
    handle = model.model[22].register_forward_pre_hook(
        lambda _module, inputs: seen.append([tuple(feature.shape) for feature in inputs[0]])
    )
    try:
        for index, channels in [(10, 256), (13, 128)]:
            layer = model.model[index]
            assert isinstance(layer, DySample)
            assert (layer.scale, layer.style, layer.groups) == (2, "lp", 4)
            assert layer.scope.in_channels == channels
            assert torch.count_nonzero(layer.scope.weight) == 0
        model.train()
        outputs = model(torch.randn(1, 3, 64, 64))
        assert seen[-1] == [(1, 64, 8, 8), (1, 128, 4, 4), (1, 256, 2, 2)]
        sum(output.square().mean() for output in outputs).backward()
        for index in (10, 13):
            grad = model.model[index].scope.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    finally:
        handle.remove()
