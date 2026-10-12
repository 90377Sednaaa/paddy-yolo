"""Standard CIoU baseline training with numerically stable FP32 validation."""

from copy import copy
import math

import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


class FP32DetectionValidator(DetectionValidator):
    """Keep AMP training, but avoid FP16 overflow during validation loss sums.

    Ultralytics forces validation to FP16 whenever trainer.amp is enabled.
    Its loss then runs outside autocast. Temporarily overriding that flag
    selects the upstream FP32 validation path without changing CIoU or BCE.
    """

    def __call__(self, trainer=None, model=None):
        if trainer is None:
            self.args.half = False
            return super().__call__(trainer=trainer, model=model)
        training_amp = trainer.amp
        try:
            trainer.amp = False
            return super().__call__(trainer=trainer, model=model)
        finally:
            trainer.amp = training_amp


def require_finite_epoch(trainer):
    """Fail rather than silently complete a replication with NaN/Inf losses."""
    if trainer.tloss is not None and not torch.isfinite(trainer.tloss).all():
        raise FloatingPointError(f"Non-finite training loss at epoch {trainer.epoch + 1}")
    bad = {key: value for key, value in trainer.metrics.items()
           if not math.isfinite(float(value))}
    if bad:
        raise FloatingPointError(f"Non-finite validation results at epoch {trainer.epoch + 1}: {bad}")


class BaselineTrainer(DetectionTrainer):
    """Unmodified YOLOv8 architecture and losses, with validation in FP32."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_fit_epoch_end", require_finite_epoch)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return FP32DetectionValidator(self.test_loader, save_dir=self.save_dir,
                                      args=copy(self.args), _callbacks=self.callbacks)
