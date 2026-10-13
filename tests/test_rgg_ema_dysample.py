"""Joint feature flow, weight averaging, and historical RGG checkpoints."""
import pytest
import torch
from torch import nn
from ultralytics.utils.torch_utils import ModelEMA

from paddy_yolo.modules import DySample, EMA, RGG, RGGBottleneck, RGGConv


def legacy_aliases(model):
    """Recreate module registrations found in previously saved checkpoints."""
    for module in list(model.modules()):
        if isinstance(module, RGGConv):
            for alias, canonical in module._module_aliases.items():
                module._modules[alias] = module._modules[canonical]
        elif isinstance(module, RGGBottleneck):
            module._modules['rgg_conv'] = module._modules['cv2']
    return model


@pytest.mark.parametrize('channels', [64, 128, 256])
@pytest.mark.parametrize('style', ['lp', 'pl'])
def test_rgg_attention_sampling_feature_flow(channels, style):
    pipeline = nn.Sequential(RGG(channels, channels, n=2), EMA(channels),
                             DySample(channels, style=style))
    # Odd sizes and a noncontiguous layout exercise reshape and sampling contracts.
    x = torch.randn(2, channels, 7, 5).transpose(2, 3).requires_grad_()
    output = pipeline(x)
    assert output.shape == (2, channels, 10, 14)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in (pipeline[0].cv1.conv.weight, pipeline[0].m[0].cv2.rep.conv1.conv.weight,
                      pipeline[1].conv3x3.weight, pipeline[2].offset.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_rgg_attention_sampling_cpu_autocast():
    pipeline = nn.Sequential(RGG(64, 64), EMA(64), DySample(64))
    x = torch.randn(2, 64, 5, 7, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = pipeline(x)
        loss = output.float().square().mean()
    loss.backward()
    assert output.shape == (2, 64, 10, 14)
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(pipeline[2].offset.weight.grad).all()


def test_rgg_aliases_do_not_duplicate_state_or_weight_averaging():
    model = RGG(64, 64, n=2)
    block = model.m[0]
    assert block.rgg_conv is block.cv2
    assert block.cv2.cv1 is block.cv2.rep
    assert block.cv2.cv2 is block.cv2.ghost1
    assert block.cv2.cv3 is block.cv2.ghost2
    assert block.cv2.fuse is block.cv2.conv
    state = model.state_dict()
    assert len({value.data_ptr() for value in state.values()}) == len(state)
    average = ModelEMA(model)
    before = {key: value.clone() for key, value in average.ema.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
        for buffer in model.buffers():
            if buffer.is_floating_point():
                buffer.add_(1)
    average.update(model)
    decay = average.decay(1)
    for key, value in average.ema.state_dict().items():
        if value.is_floating_point():
            torch.testing.assert_close(value, before[key] * decay + model.state_dict()[key] * (1 - decay))


def test_rgg_legacy_state_dict_loads_strictly():
    historical = legacy_aliases(RGG(64, 64, n=2)).eval()
    restored = RGG(64, 64, n=2).eval()
    restored.load_state_dict(historical.state_dict(), strict=True)
    assert len(restored.state_dict()) < len(historical.state_dict())
    image = torch.randn(2, 64, 5, 7)
    with torch.no_grad():
        torch.testing.assert_close(restored(image), historical(image))


def test_rgg_legacy_whole_model_checkpoint_normalizes_aliases(tmp_path):
    historical = legacy_aliases(RGG(64, 64, n=2)).eval()
    checkpoint = tmp_path / 'legacy_rgg.pt'
    image = torch.randn(2, 64, 5, 7)
    with torch.no_grad():
        expected = historical(image)
    torch.save({'model': historical}, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)['model']
    state = restored.state_dict()
    assert len({value.data_ptr() for value in state.values()}) == len(state)
    assert len(state) < len(historical.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected)


def test_rgg_deploy_preserves_attention_sampling_output():
    pipeline = nn.Sequential(RGG(64, 64), EMA(64), DySample(64)).eval()
    image = torch.randn(2, 64, 5, 7)
    with torch.no_grad():
        expected = pipeline(image)
        torch.testing.assert_close(pipeline[0].forward_split(image), pipeline[0](image))
        pipeline[0].switch_to_deploy()
        torch.testing.assert_close(pipeline(image), expected, rtol=2e-4, atol=2e-5)
