"""DySample: Ultra-lightweight Dynamic Upsampler for Object Detection.

Reference:
    Liu et al., "DySample: A Lightweight Dynamic Upsampler for Object Detection and Segmentation",
    ICCV 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Ensure groups divide in_channels; fallback to 1 if not divisible or in_channels < groups
        if groups > in_channels or in_channels % groups != 0:
            groups = 1
        self.groups = groups
        self.dyscope = dyscope

        out_channels = 2 * groups * scale * scale if style == "lp" else 2 * scale * scale
        self.offset = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=True)
            nn.init.constant_(self.scope.weight, 0)
            nn.init.constant_(self.scope.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        g = self.groups

        offset = self.offset(x)
        if self.dyscope:
            scope = torch.sigmoid(self.scope(x)) * 0.5
            offset = offset * scope

        # Normalized base meshgrid coordinates in [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(
                -1.0 + 1.0 / (s * h),
                1.0 - 1.0 / (s * h),
                s * h,
                device=x.device,
                dtype=x.dtype,
            ),
            torch.linspace(
                -1.0 + 1.0 / (s * w),
                1.0 - 1.0 / (s * w),
                s * w,
                device=x.device,
                dtype=x.dtype,
            ),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, s*h, s*w)

        if self.style == "lp":
            offset = F.pixel_shuffle(offset, s)  # (B, 2*g, s*h, s*w)
            offset_x = offset[:, 0::2] / w
            offset_y = offset[:, 1::2] / h
            offset = torch.cat([offset_x, offset_y], dim=1)  # (B, 2*g, s*h, s*w)

            x_g = x.view(b * g, c // g, h, w)
            bg = base_grid.repeat(b * g, 1, 1, 1)  # (b*g, 2, s*h, s*w)
            off = offset.view(b * g, 2, s * h, s * w)
            sample_grid = (bg + off).permute(0, 2, 3, 1)  # (b*g, s*h, s*w, 2)

            out = F.grid_sample(
                x_g,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            return out.view(b, c, s * h, s * w)
        else:
            offset = F.pixel_shuffle(offset, s)
            offset[:, 0:1] /= w
            offset[:, 1:2] /= h
            sample_grid = (base_grid + offset).permute(0, 2, 3, 1)
            return F.grid_sample(
                x,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
