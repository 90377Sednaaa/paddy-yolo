"""PaddyTrainer: Custom Ultralytics DetectionTrainer integration for Paddy-YOLO.

Integrates custom Paddy-YOLO modules (RGG, GhostConv, RepConv) and
Dynamic Inner-MPDIoU (DIMIoU) loss into the Ultralytics training engine.
"""

import copy
from typing import Any, Dict, Optional

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from paddy_yolo.loss.paddy_loss import PaddyDetectionLoss
from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.engine.baseline import BaselineTrainer


class PaddyCriterionInitializer:
    """Picklable callable criterion initializer for Ultralytics DetectionModel."""

    def __init__(
        self,
        model: torch.nn.Module,
        strategy: str = "H",
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.75,
        fixed_ratio=None,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.fixed_ratio = fixed_ratio

    def __call__(self, *args, **kwargs) -> PaddyDetectionLoss:
        return PaddyDetectionLoss(
            self.model,
            strategy=self.strategy,
            focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha,
            fixed_ratio=getattr(self, "fixed_ratio", None),
        )


class PaddyTrainer(BaselineTrainer):
    """Detection trainer customized for Paddy-YOLO.

    Features:
    1. Automatic registration of Paddy-YOLO architectural modules into Ultralytics.
    2. Dynamic Inner-MPDIoU (DIMIoU Strategy H) loss substitution for bounding box regression.
    3. ClassWeightedFocalLoss support to address extreme agricultural pest class imbalance.
    4. Seamless preservation of PaddyDetectionLoss for both training and validation phases.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[Dict[str, Any]] = None,
        _callbacks=None,
    ) -> None:
        """Initialize PaddyTrainer.

        Args:
            cfg: Configuration path or dict. Defaults to DEFAULT_CFG.
            overrides: Configuration overrides dict. Defaults to None.
            _callbacks: Callback dict or list. Defaults to None.
        """
        # Ensure custom modules are registered before any configuration loading
        register_paddy_modules()

        # Extract strategy and focal params if present in overrides to avoid Ultralytics CLI arg validation error
        clean_overrides = copy.deepcopy(overrides) if overrides else {}
        self.strategy = clean_overrides.pop("strategy", "H")
        self.fixed_ratio = clean_overrides.pop("fixed_ratio", None)
        self.focal_gamma = float(clean_overrides.pop("focal_gamma", 0.0))
        self.focal_alpha = float(clean_overrides.pop("focal_alpha", 0.75))

        super().__init__(cfg=cfg, overrides=clean_overrides, _callbacks=_callbacks)

        # Attach validation callback to guarantee PaddyDetectionLoss is used during validation
        self.add_callback("on_val_start", self._on_val_start)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Construct YOLO detection model and configure PaddyDetectionLoss criterion.

        Args:
            cfg: Model architecture config file path or dict.
            weights: Optional pretrained weights path.
            verbose: Whether to log model summary.

        Returns:
            DetectionModel: Configured YOLO detection model.
        """
        register_paddy_modules()
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        strategy = getattr(self, "strategy", "H")
        focal_gamma = getattr(self, "focal_gamma", 0.0)
        focal_alpha = getattr(self, "focal_alpha", 0.75)
        model.init_criterion = PaddyCriterionInitializer(
            model, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
        )
        return model

    def setup_model(self):
        """Ensure Paddy-YOLO modules are registered and init_criterion is bound."""
        register_paddy_modules()
        ckpt = super().setup_model()
        if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
            strategy = getattr(self, "strategy", "H")
            focal_gamma = getattr(self, "focal_gamma", 0.0)
            focal_alpha = getattr(self, "focal_alpha", 0.75)
            self.model.init_criterion = PaddyCriterionInitializer(
                self.model, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
            )
        return ckpt

    def _on_val_start(self, validator=None) -> None:
        """Callback ensuring PaddyDetectionLoss is used during validation loss computation."""
        strategy = getattr(self, "strategy", "H")
        focal_gamma = getattr(self, "focal_gamma", 0.0)
        focal_alpha = getattr(self, "focal_alpha", 0.75)

        # Configure trainer model
        if hasattr(self, "model") and self.model is not None:
            model = unwrap_model(self.model)
            model.init_criterion = PaddyCriterionInitializer(
                model, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
            )
            model.criterion = PaddyDetectionLoss(
                model, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
            )

        # Configure EMA model if present
        if (
            hasattr(self, "ema")
            and self.ema is not None
            and hasattr(self.ema, "ema")
            and self.ema.ema is not None
        ):
            self.ema.ema.init_criterion = PaddyCriterionInitializer(
                self.ema.ema, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
            )
            self.ema.ema.criterion = PaddyDetectionLoss(
                self.ema.ema, strategy=strategy, focal_gamma=focal_gamma, focal_alpha=focal_alpha, fixed_ratio=getattr(self, "fixed_ratio", None)
            )

    def validate(self):
        """Run validation ensuring PaddyDetectionLoss is applied to the evaluation model."""
        self._on_val_start(getattr(self, "validator", None))
        return super().validate()

    def get_validator(self):
        """Return DetectionValidator configured for Paddy-YOLO detection metrics and losses."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = super().get_validator()
        return validator
