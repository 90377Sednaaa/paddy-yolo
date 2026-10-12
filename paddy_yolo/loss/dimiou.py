"""Dynamic Inner-MPDIoU (DIMIoU) loss function with Strategy H dynamic ratio scheduling.

Reference:
    Zheng et al., "Paddy-YOLO: An accurate method for rice pest detection",
    Computers and Electronics in Agriculture 238 (2025) 110777.
    Section 2.2.2: IMPDIoU and DIMIoU loss (Equations 1-23).
"""

import math
import torch


def get_dynamic_ratio(iou: torch.Tensor, strategy: str = "H") -> torch.Tensor:
    """Compute dynamic auxiliary scaling ratio based on current IoU.

    Implements Strategies A through L from Section 2.2.2 (Equations 12-23).
    All strategies map IoU in [0, 1] to ratio values in [0.5, 1.5].

    Args:
        iou (torch.Tensor): Current base IoU values, typically in [0.0, 1.0].
        strategy (str): Strategy identifier ('A' through 'L', case-insensitive). Default is 'H'.

    Returns:
        torch.Tensor: Auxiliary scaling ratio tensor of the same shape as iou.

    Raises:
        ValueError: If an unrecognized strategy identifier is provided.
    """
    strat = strategy.upper()
    pi = math.pi

    if strat == "A":
        # Eq. 12: ratio = 1.5 - IoU
        return 1.5 - iou
    elif strat == "B":
        # Eq. 13: ratio = 0.5 * cos(IoU * pi) + 1.0
        return 0.5 * torch.cos(iou * pi) + 1.0
    elif strat == "C":
        # Eq. 14: ratio = 1.0 / (1 + exp((IoU - 0.5) * 10)) + 0.5
        return 1.0 / (1.0 + torch.exp((iou - 0.5) * 10.0)) + 0.5
    elif strat == "D":
        # Eq. 15: Piecewise cosine peaking at low-to-mid IoU
        mid = 0.5 * torch.cos(1.25 * (iou - 0.1) * pi) + 1.0
        return torch.where((iou >= 0.1) & (iou <= 0.9), mid, torch.full_like(iou, 0.5))
    elif strat == "E":
        # Eq. 16: ratio = cos((IoU - 0.5) * pi) + 0.5
        return torch.cos((iou - 0.5) * pi) + 0.5
    elif strat == "F":
        # Eq. 17: Binary step: 1.5 if IoU < 0.5 else 0.5
        return torch.where(iou < 0.5, torch.full_like(iou, 1.5), torch.full_like(iou, 0.5))
    elif strat == "G":
        # Eq. 18: ratio = IoU + 0.5
        return iou + 0.5
    elif strat == "H":
        # Eq. 19: ratio = 0.5 * cos((1 - IoU) * pi) + 1.0 (Default Strategy in Paddy-YOLO)
        return 0.5 * torch.cos((1.0 - iou) * pi) + 1.0
    elif strat == "I":
        # Eq. 20: ratio = 1.0 / (1 + exp((0.5 - IoU) * 10)) + 0.5
        return 1.0 / (1.0 + torch.exp((0.5 - iou) * 10.0)) + 0.5
    elif strat == "J":
        # Eq. 21: Piecewise cosine rising to peak at high IoU
        mid = 0.5 * torch.cos(1.25 * (0.9 - iou) * pi) + 1.0
        return torch.where((iou >= 0.1) & (iou <= 0.9), mid, torch.full_like(iou, 0.5))
    elif strat == "K":
        # Eq. 22: ratio = cos((IoU + 0.5) * pi) + 1.5
        return torch.cos((iou + 0.5) * pi) + 1.5
    elif strat == "L":
        # Eq. 23: Binary step: 0.5 if IoU < 0.5 else 1.5
        return torch.where(iou < 0.5, torch.full_like(iou, 0.5), torch.full_like(iou, 1.5))
    else:
        raise ValueError(
            f"Unknown DIMIoU strategy '{strategy}'. Supported strategies: 'A' through 'L'."
        )


def dimiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    strategy: str = "H",
    eps: float = 1e-7,
    ratio: float | None = None,
) -> torch.Tensor:
    """Compute Dynamic Inner-MPDIoU (DIMIoU) metric values.

    Combines:
    1. Base IoU calculation.
    2. Dynamic ratio adjustment conditioned on IoU (default Strategy H).
    3. InnerIoU scaling by ratio from box centers.
    4. MPDIoU normalized corner distances penalty.

    DIMIoU = InnerIoU - (d1^2 + d2^2) / C^2

    Normalization assumption: C is the smallest enclosing box diagonal.
    Paper Eq. (2) uses the input image diagonal, but Eqs. (4)-(5) use
    w^2+h^2 without specifying which box. This implementation's choice
    must not be described as verified agreement with the authors' code.

    Args:
        box1 (torch.Tensor): Bounding box tensor of shape (..., 4).
        box2 (torch.Tensor): Bounding box tensor of shape (..., 4).
        xywh (bool): If True, coordinates are (x_center, y_center, w, h).
                     If False, coordinates are (x1, y1, x2, y2). Default is False.
        strategy (str): Dynamic ratio scheduling strategy ('A'-'L'). Default is 'H'.
        eps (float): Small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        torch.Tensor: DIMIoU metric values of shape (..., 1).
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_half, h1_half = w1 / 2, h1 / 2
        w2_half, h2_half = w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_half, x1 + w1_half
        b1_y1, b1_y2 = y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2 = x2 - w2_half, x2 + w2_half
        b2_y1, b2_y2 = y2 - h2_half, y2 + h2_half
        b1_xc, b1_yc = x1, y1
        b2_xc, b2_yc = x2, y2
        w1 = w1.clamp(min=0)
        h1 = h1.clamp(min=0)
        w2 = w2.clamp(min=0)
        h2 = h2.clamp(min=0)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = (b1_x2 - b1_x1).clamp(min=0)
        h1 = (b1_y2 - b1_y1).clamp(min=0)
        w2 = (b2_x2 - b2_x1).clamp(min=0)
        h2 = (b2_y2 - b2_y1).clamp(min=0)
        b1_xc = (b1_x1 + b1_x2) / 2
        b1_yc = (b1_y1 + b1_y2) / 2
        b2_xc = (b2_x1 + b2_x2) / 2
        b2_yc = (b2_y1 + b2_y2) / 2

    # 1. Base IoU calculation
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = torch.clamp(inter / union, 0.0, 1.0)

    # 2. Dynamic ratio adjustment
    if ratio is None:
        ratio = get_dynamic_ratio(iou, strategy=strategy)
    else:
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("IMPDIoU requires a finite positive fixed ratio")
        ratio = torch.full_like(iou, ratio)

    # 3. InnerIoU scaled boxes around center coordinates
    w1_r_half = (w1 * ratio) / 2
    h1_r_half = (h1 * ratio) / 2
    w2_r_half = (w2 * ratio) / 2
    h2_r_half = (h2 * ratio) / 2

    b1_x1_r, b1_x2_r = b1_xc - w1_r_half, b1_xc + w1_r_half
    b1_y1_r, b1_y2_r = b1_yc - h1_r_half, b1_yc + h1_r_half
    b2_x1_r, b2_x2_r = b2_xc - w2_r_half, b2_xc + w2_r_half
    b2_y1_r, b2_y2_r = b2_yc - h2_r_half, b2_yc + h2_r_half

    inter_r = (b1_x2_r.minimum(b2_x2_r) - b1_x1_r.maximum(b2_x1_r)).clamp(min=0) * (
        b1_y2_r.minimum(b2_y2_r) - b1_y1_r.maximum(b2_y1_r)
    ).clamp(min=0)
    union_r = (w1 * ratio) * (h1 * ratio) + (w2 * ratio) * (h2 * ratio) - inter_r + eps
    inner_iou = inter_r / union_r

    # 4. MPDIoU normalized corner distances
    d1_sq = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2_sq = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)

    # Smallest enclosing box dimensions
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    corner_penalty = (d1_sq + d2_sq) / c2

    return inner_iou - corner_penalty


def dimiou_loss(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    strategy: str = "H",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Dynamic Inner-MPDIoU (DIMIoU) regression loss.

    L_DIMIoU = 1.0 - DIMIoU
             = 1.0 - InnerIoU + (d1^2 + d2^2) / C^2

    Args:
        box1 (torch.Tensor): Bounding box tensor of shape (..., 4).
        box2 (torch.Tensor): Bounding box tensor of shape (..., 4).
        xywh (bool): If True, coordinates are (x_center, y_center, w, h).
                     If False, coordinates are (x1, y1, x2, y2). Default is False.
        strategy (str): Dynamic ratio scheduling strategy ('A'-'L'). Default is 'H'.
        eps (float): Small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        torch.Tensor: DIMIoU loss tensor of shape (..., 1).
    """
    return 1.0 - dimiou(box1, box2, xywh=xywh, strategy=strategy, eps=eps)


def impdiou(box1, box2, xywh=False, ratio=1.3, eps=1e-7):
    """Fixed-ratio IMPDIoU, Eqs. (4)-(5); paper's R2000 ratio is 1.3.

    Retains the disclosed enclosing-diagonal normalization assumption.
    """
    return dimiou(box1, box2, xywh=xywh, eps=eps, ratio=ratio)


def impdiou_loss(box1, box2, xywh=False, ratio=1.3, eps=1e-7):
    """Fixed-ratio Inner-MPDIoU loss, rather than dynamic Strategy H."""
    return 1.0 - impdiou(box1, box2, xywh=xywh, ratio=ratio, eps=eps)
