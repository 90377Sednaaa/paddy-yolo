"""Regression checks for the numerical failures found in the real Kaggle run."""
from types import SimpleNamespace
import pytest
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from paddy_yolo.engine.baseline import FP32DetectionValidator, require_finite_epoch


@pytest.mark.parametrize("raises", [False, True])
def test_fp32_validation_preserves_training_amp(monkeypatch, raises):
    trainer = SimpleNamespace(amp=torch.tensor(True))
    original = trainer.amp
    def validate(self, trainer=None, model=None):
        assert trainer.amp is False
        if raises:
            raise RuntimeError("validator failure")
        return {"val/cls_loss": 1.0}
    monkeypatch.setattr(DetectionValidator, "__call__", validate)
    validator = FP32DetectionValidator.__new__(FP32DetectionValidator)
    if raises:
        with pytest.raises(RuntimeError):
            validator(trainer=trainer)
    else:
        assert validator(trainer=trainer) == {"val/cls_loss": 1.0}
    assert trainer.amp is original


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_validation_is_rejected(bad):
    trainer = SimpleNamespace(tloss=torch.ones(3), epoch=0, metrics={"val/cls_loss": bad})
    with pytest.raises(FloatingPointError, match="Non-finite validation"):
        require_finite_epoch(trainer)


def test_nonfinite_training_is_rejected():
    trainer = SimpleNamespace(tloss=torch.tensor([1., float("nan"), 1.]), epoch=0, metrics={})
    with pytest.raises(FloatingPointError, match="Non-finite training"):
        require_finite_epoch(trainer)


def test_finite_epoch_is_accepted():
    require_finite_epoch(SimpleNamespace(tloss=torch.ones(3), epoch=0, metrics={"val/cls_loss": 2.0}))


@pytest.mark.parametrize("history,exception", [
    ("epoch,val/cls_loss\n1,1.0\n2,inf\n", FloatingPointError),
    ("epoch,val/cls_loss\n1,1.0\n", ValueError),
])
def test_incomplete_or_nonfinite_history_is_not_publishable(tmp_path, history, exception):
    from run_baseline_crossval import check_training_history
    path = tmp_path / "results.csv"
    path.write_text(history)
    with pytest.raises(exception):
        check_training_history(path, expected_epochs=2)


def test_finite_complete_history_is_publishable(tmp_path):
    from run_baseline_crossval import check_training_history
    path = tmp_path / "results.csv"
    path.write_text("epoch,val/cls_loss\n1,1.0\n2,2.0\n")
    assert check_training_history(path, expected_epochs=2) == {"epochs": 2, "all_values_finite": True}
