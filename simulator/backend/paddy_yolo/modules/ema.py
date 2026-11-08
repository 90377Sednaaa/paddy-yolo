"""Efficient Multi-Scale Attention (EMA) Module with Cross-Spatial Learning.

Reference:
Ouyang, D., He, S., Zhang, G., Luo, M., Guo, H., Zhan, J., & Huang, Z. (2023).
"Efficient Multi-Scale Attention Module with Cross-Spatial Learning."
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023).
Algorithm reference: https://github.com/YOLOonMe/EMA-attention-module
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (EMA) Module.

    Splits input channels into sub-groups to learn parallel multi-scale features,
    encodes directional 1D spatial coordinate statistics, and establishes cross-spatial
    pixel interactions across parallel sub-networks without channel dimensionality reduction.

    Args:
        channels: Number of input/output channels. Also accepts 'c1' as alias.
        factor: Number of channel grouping partitions (default: 8).
    """

    def __init__(
        self,
        channels: int | None = None,
        c1: int | None = None,
        c2: int | None = None,
        factor: int = 8,
        groups: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # Flexible channel resolution for standalone and Ultralytics parse_model calls
        if channels is not None:
            c = channels
        elif c1 is not None:
            c = c1
        elif len(args) > 0 and isinstance(args[0], int):
            c = args[0]
        else:
            c = kwargs.get("ch", 32)

        self.channels = c
        self.groups = groups if groups is not None else factor

        # Graceful adjustment if channel count is not divisible by groups
        if c % self.groups != 0:
            for f in [8, 4, 2, 1]:
                if c % f == 0:
                    self.groups = f
                    break

        group_channels = c // self.groups
        assert group_channels > 0, f"Channel count {c} must be >= groups {self.groups}"

        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(
            group_channels, group_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            group_channels, group_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying cross-spatial multi-scale attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Attention-reweighted tensor of same shape (batch, channels, height, width).
        """
        batch, channels, height, width = x.shape
        groups = getattr(self, "groups", 8)
        grouped = x.reshape(batch * groups, channels // groups, height, width)

        # 1D directional coordinate statistics (Height and Width context)
        height_context = grouped.mean(dim=3, keepdim=True)
        width_context = grouped.mean(dim=2, keepdim=True).transpose(2, 3)

        # 1x1 conv encoding across concatenated coordinate dimensions
        context = self.conv1x1(torch.cat((height_context, width_context), dim=2))
        height_gate, width_gate = context.split((height, width), dim=2)

        # Multi-scale parallel sub-networks
        gated = self.gn(grouped * height_gate.sigmoid() * width_gate.transpose(2, 3).sigmoid())
        spatial = self.conv3x3(grouped)

        # Cross-spatial learning interaction
        gated_query = gated.mean((2, 3)).unsqueeze(1).softmax(dim=-1)
        spatial_query = spatial.mean((2, 3)).unsqueeze(1).softmax(dim=-1)

        weights = (gated_query @ spatial.flatten(2) + spatial_query @ gated.flatten(2))
        weights = weights.reshape(batch * groups, 1, height, width).sigmoid()

        # Attention reweighting
        return (grouped * weights).reshape(batch, channels, height, width)
