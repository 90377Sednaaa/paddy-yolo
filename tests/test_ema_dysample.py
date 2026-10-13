"""Numerics, connectivity, transfer and checkpoint checks for the enhanced model."""
import torch
from copy import deepcopy
from torch.nn import functional as F
from ultralytics import YOLO
import pytest

import paddy_yolo
from paddy_yolo.modules import EMA, DySample
from paddy_yolo.engine.pretrained import EMA_DYSAMPLE_LAYER_MAP, transfer_yolov8_weights

ARCHITECTURE = "configs/paddy-yolo-ema-dysample.yaml"


def test_ema_preserves_shape_and_backpropagates():
    module = EMA(64)
    x = torch.randn(2, 64, 5, 7, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("style", ["lp", "pl"])
@pytest.mark.parametrize("dyscope", [False, True])
def test_dysample_learned_offsets_receive_gradients(style, dyscope):
    module = DySample(16, style=style, dyscope=dyscope)
    x = torch.randn(2, 16, 5, 7, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 16, 10, 14)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert module.offset.weight.grad.abs().sum() > 0
    if dyscope:
        assert module.scope.weight.grad is not None


@pytest.mark.parametrize("style", ["lp", "pl"])
def test_dysample_zero_offsets_match_bilinear_interpolation(style):
    module = DySample(16, style=style)
    torch.nn.init.zeros_(module.offset.weight)
    torch.nn.init.zeros_(module.offset.bias)
    x = torch.randn(2, 16, 3, 5)
    reference = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(module(x), reference, atol=2e-6, rtol=2e-6)


def test_attention_rejects_invalid_channel_groups():
    with pytest.raises(ValueError):
        EMA(63)
    with pytest.raises(ValueError):
        DySample(63)
    with pytest.raises(ValueError):
        DySample(6, groups=2, style="pl")


def test_enhanced_graph_and_detection_shapes():
    model = YOLO(ARCHITECTURE).model
    layers = model.model
    assert [layer.i for layer in layers if isinstance(layer, EMA)] == [5, 8, 11]
    assert [layer.i for layer in layers if isinstance(layer, DySample)] == [13, 16]
    assert layers[14].f == [-1, 8]
    assert layers[17].f == [-1, 5]
    assert layers[20].f == [-1, 15]
    assert layers[23].f == [-1, 12]
    assert layers[25].f == [18, 21, 24]
    assert model.stride.tolist() == [8, 16, 32]
    assert layers[25].nc == 16
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 640, 640))[0]
    assert output.shape == (1, 20, 8400)


def test_enhanced_loss_updates_attention_and_sampling_offsets():
    from paddy_yolo.loss import PaddyDetectionLoss

    model = YOLO(ARCHITECTURE).model
    model.train()
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0, 1]),
             "cls": torch.tensor([[0.], [1.]]),
             "bboxes": torch.tensor([[.5, .5, .4, .4], [.5, .5, .3, .3]])}
    loss, items = PaddyDetectionLoss(model)(model(batch["img"]), batch)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    loss.backward()
    for layer in model.model:
        if isinstance(layer, EMA):
            assert layer.conv3x3.weight.grad.abs().sum() > 0
        if isinstance(layer, DySample):
            assert layer.offset.weight.grad.abs().sum() > 0


def test_pretraining_maps_shifted_layers_and_preserves_new_modules(monkeypatch):
    import paddy_yolo.engine.pretrained as pretrained

    target = YOLO(ARCHITECTURE).model
    source = YOLO("yolov8n.yaml")
    with torch.no_grad():
        for i, layer in enumerate(source.model.model):
            for value in layer.state_dict().values():
                if value.is_floating_point():
                    value.fill_(i + 1)
    before = {key: value.clone() for key, value in target.state_dict().items()
              if key.startswith(tuple(f"model.{i}." for i in (5, 8, 11, 13, 16)))}
    monkeypatch.setattr(pretrained, "YOLO", lambda weights: source)
    report = transfer_yolov8_weights(target, "yolov8n.pt", ARCHITECTURE)
    for src, dst in ((5, 6), (7, 9), (9, 12), (12, 15), (16, 19), (19, 22), (21, 24)):
        target_layer = target.model[dst]
        source_layer = source.model.model[src]
        target_key = "conv.weight" if hasattr(target_layer, "conv") else "cv1.conv.weight"
        torch.testing.assert_close(target_layer.state_dict()[target_key], source_layer.state_dict()[target_key])
    assert report["layer_map"] == EMA_DYSAMPLE_LAYER_MAP
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_enhanced_checkpoint_round_trip(tmp_path):
    model = YOLO(ARCHITECTURE)
    model.model.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model.model(image)[0]
    checkpoint = tmp_path / "enhanced.pt"
    # Raw YAML models have no training checkpoint metadata in Ultralytics 8.3.18.
    # Use the same model checkpoint format as the transfer/training pipeline.
    torch.save({"model": deepcopy(model.model).half()}, checkpoint)
    reloaded = YOLO(checkpoint).model.eval()
    with torch.no_grad():
        actual = reloaded(image)[0]
    # YOLO.save stores half-precision weights, so permit checkpoint quantization.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert sum(isinstance(layer, EMA) for layer in reloaded.model) == 3
    assert sum(isinstance(layer, DySample) for layer in reloaded.model) == 2
