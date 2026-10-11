"""Detection loss functions for Paddy-YOLO incorporating DIMIoU regression.

Reference:
    Zheng et al., "Paddy-YOLO: An accurate method for rice pest detection",
    Computers and Electronics in Agriculture 238 (2025) 110777.
    Section 2.2.2: IMPDIoU and DIMIoU loss with Strategy H scheduling.
"""

from typing import Tuple

import torch
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss, bbox2dist, v8DetectionLoss

from paddy_yolo.loss.dimiou import dimiou


class PaddyBboxLoss(BboxLoss):
    """Criterion for bounding box regression using Dynamic Inner-MPDIoU (DIMIoU).

    Subclasses Ultralytics `BboxLoss` and replaces CIoU with DIMIoU
    (Strategy H by default) for bounding box regression.
    """

    def __init__(self, reg_max: int = 16, strategy: str = "H") -> None:
        """Initialize PaddyBboxLoss.

        Args:
            reg_max (int): Maximum regularization value for DFL. Default is 16.
            strategy (str): Dynamic ratio scheduling strategy ('A'-'L'). Default is 'H'.
        """
        super().__init__(reg_max)
        self.strategy = strategy

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor | float,
        fg_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU loss with DIMIoU and DFL loss.

        Args:
            pred_dist: Predicted distribution for DFL of shape (b, anchors, 4 * reg_max).
            pred_bboxes: Predicted bounding boxes of shape (b, anchors, 4) in xyxy format.
            anchor_points: Anchor points of shape (anchors, 2).
            target_bboxes: Target bounding boxes of shape (b, anchors, 4) in xyxy format.
            target_scores: Target scores of shape (b, anchors, nc).
            target_scores_sum: Sum of target scores across foreground objects.
            fg_mask: Foreground mask of shape (b, anchors).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss_iou, loss_dfl).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = dimiou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            strategy=self.strategy,
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class ClassWeightedFocalLoss(torch.nn.Module):
    """Focal Loss with class-frequency balancing for extreme class imbalance.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: float = 0.75,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if self.gamma > 0.0:
            pred_prob = pred.sigmoid()
            p_t = target * pred_prob + (1.0 - target) * (1.0 - pred_prob)
            modulating_factor = (1.0 - p_t) ** self.gamma
            loss = bce * modulating_factor
        else:
            loss = bce

        if self.alpha > 0.0:
            alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
            loss = loss * alpha_factor

        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, 1, -1)

        return loss


class PaddyDetectionLoss(v8DetectionLoss):
    """Detection loss criterion for Paddy-YOLO using DIMIoU regression.

    Subclasses Ultralytics `v8DetectionLoss` and initializes `self.bbox_loss`
    with `PaddyBboxLoss` using Strategy H by default, and optionally employs
    ClassWeightedFocalLoss to handle extreme class imbalance.
    """

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        strategy: str = "H",
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.75,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """Initialize PaddyDetectionLoss.

        Args:
            model: YOLO detection model.
            tal_topk (int): Number of top candidates for TaskAlignedAssigner. Default is 10.
            strategy (str): Dynamic ratio scheduling strategy for DIMIoU. Default is 'H'.
            focal_gamma (float): Focusing parameter for Focal Loss. Default is 1.5 (0 to disable).
            focal_alpha (float): Positive/negative balance parameter for Focal Loss. Default is 0.75.
            class_weights (Tensor | None): Optional per-class weight tensor of shape (nc,).
        """
        # Ensure model has args with box, cls, dfl gains formatted for attribute access
        if not hasattr(model, "args") or model.args is None:
            model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        elif isinstance(model.args, dict):
            args_dict = dict(box=7.5, cls=0.5, dfl=1.5)
            args_dict.update(model.args)
            model.args = IterableSimpleNamespace(**args_dict)
        elif not hasattr(model.args, "box"):
            try:
                model.args.box = 7.5
                model.args.cls = getattr(model.args, "cls", 0.5)
                model.args.dfl = getattr(model.args, "dfl", 1.5)
            except Exception:
                args_dict = dict(box=7.5, cls=0.5, dfl=1.5)
                if hasattr(model.args, "__dict__"):
                    args_dict.update(model.args.__dict__)
                model.args = IterableSimpleNamespace(**args_dict)

        super().__init__(model, tal_topk=tal_topk)

        # Ensure self.hyp supports attribute access
        if isinstance(self.hyp, dict):
            hyp_dict = dict(box=7.5, cls=0.5, dfl=1.5)
            hyp_dict.update(self.hyp)
            self.hyp = IterableSimpleNamespace(**hyp_dict)
        elif not hasattr(self.hyp, "box"):
            self.hyp = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        self.strategy = strategy
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.m = model.model[-1] if hasattr(model, "model") else None
        reg_max = getattr(self.m, "reg_max", getattr(self, "reg_max", 16))
        self.bbox_loss = PaddyBboxLoss(reg_max, strategy=strategy).to(self.device)

        if focal_gamma > 0.0:
            self.bce = ClassWeightedFocalLoss(
                gamma=focal_gamma,
                alpha=focal_alpha,
                class_weights=class_weights,
            ).to(self.device)

    def forward(self, preds, batch):
        """Forward pass alias for __call__."""
        return self(preds, batch)
