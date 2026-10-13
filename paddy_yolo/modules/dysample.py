"""DySample: Ultra-lightweight Dynamic Upsampler for Object Detection.

Reference:
    Liu et al., "DySample: A Lightweight Dynamic Upsampler for Object Detection and Segmentation",
    ICCV 2023.
    Official implementation: https://github.com/tiny-smart/dysample
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.001, bias: float = 0.0) -> None:
    """Initialize module weights with normal distribution and bias with constant."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _constant_init(module: nn.Module, val: float = 0.0, bias: float = 0.0) -> None:
    """Initialize module weights with constant and bias with constant."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """Dynamic Point-based Upsampler.

    Args:
        in_channels: Number of input feature channels.
        scale: Spatial upsampling factor (typically 2).
        style: 'lp' (linear projection) or 'pl' (point-wise convolution). Default: 'lp'.
        groups: Number of channel groups for independent offset sampling. Default: 4.
        dyscope: Whether to use dynamic scope adjustment. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        assert style in ["lp", "pl"], f"Unsupported style '{style}'. Choose 'lp' or 'pl'."
        assert scale in [2, 4], f"Unsupported scale factor {scale}. Choose 2 or 4."

        self.scale = scale
        self.style = style

        # Ensure groups divide in_channels; fallback gracefully to 1 if not divisible
        if groups > in_channels or in_channels % groups != 0:
            groups = 1
        self.groups = groups
        self.dyscope = dyscope

        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0, (
                f"For style 'pl', in_channels ({in_channels}) must be divisible by scale^2 ({scale**2})."
            )
            conv_in = in_channels // (scale**2)
            conv_out = 2 * groups
        else:
            conv_in = in_channels
            conv_out = 2 * groups * (scale**2)

        self.offset = nn.Conv2d(conv_in, conv_out, 1, bias=True)
        _normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = nn.Conv2d(conv_in, conv_out, 1, bias=False)
            _constant_init(self.scope, val=0.0)

        # Buffer for subpixel initial sampling positions (ICCV 2023 formulation)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        """Compute base subpixel offset positions within each grid cell."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Perform dynamic point sampling via grid_sample.

        Robust to arbitrary memory formats (channels_last), non-contiguous tensors,
        and mixed precision (autocast/half).
        """
        b, _, h, w = offset.shape
        s = self.scale
        g = self.groups

        # 1. Base grid coordinate generation at low resolution (H, W)
        offset = offset.reshape(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(dtype=x.dtype, device=x.device)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0

        # 2. Pixel-shuffle coordinates to high resolution (s*H, s*W)
        coords = (
            F.pixel_shuffle(coords.reshape(b, -1, h, w), s)
            .reshape(b, 2, -1, s * h, s * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )  # shape: (b * g, s * h, s * w, 2)

        # 3. Dynamic point sampling
        # Enforce contiguous standard format for x to ensure grid_sample and reshape safety
        x_reshaped = x.contiguous().reshape(b * g, -1, h, w)
        coords = coords.to(dtype=x.dtype, device=x.device)

        out = F.grid_sample(
            x_reshaped,
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return out.reshape(b, -1, s * h, s * w)

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        """Linear projection forward pass."""
        init_pos = self.init_pos.to(dtype=x.dtype, device=x.device)
        if hasattr(self, "scope"):
            offset = self.offset(x) * torch.sigmoid(self.scope(x)) * 0.5 + init_pos
        else:
            offset = self.offset(x) * 0.25 + init_pos
        return self.sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        """Point-wise / pixel-shuffle convolution forward pass."""
        x_ = F.pixel_shuffle(x, self.scale)
        init_pos = self.init_pos.to(dtype=x.dtype, device=x.device)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(self.offset(x_) * torch.sigmoid(self.scope(x_)), self.scale) * 0.5
                + init_pos
            )
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + init_pos
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)
