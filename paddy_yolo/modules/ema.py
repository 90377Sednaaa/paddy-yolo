"""Efficient Multi-Scale Attention (EMA) Module with Cross-Spatial Learning.

Reference:
Ouyang, D., He, S., Zhang, G., Luo, M., Guo, H., Zhan, J., & Huang, Z. (2023).
"Efficient Multi-Scale Attention Module with Cross-Spatial Learning."
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023).
https://doi.org/10.1109/ICASSP49357.2023.10096516

Key characteristics:
- Multi-scale parallel sub-network encoding (1x1 and 3x3 parallel branches)
- Direction-aware 2D spatial coordinate pooling (X and Y pooling)
- Cross-spatial pixel interaction without channel dimensionality reduction
- Near-zero parameter and FLOP overhead
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        self.groups = factor

        # Graceful adjustment if channel count is not divisible by factor
        if c % self.groups != 0:
            for f in [8, 4, 2, 1]:
                if c % f == 0:
                    self.groups = f
                    break

        group_channels = c // self.groups
        assert group_channels > 0, f"Channel count {c} must be >= groups {self.groups}"

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

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
        b, c, h, w = x.size()
        g = self.groups
        gc = c // g

        # 1. Feature grouping: reshape channels into batch dimension
        group_x = x.reshape(b * g, gc, h, w)

        # 2. 1D directional coordinate pooling (Height & Width)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 3. Concatenate along spatial dimension & 1x1 conv encoding
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 4. Multi-scale parallel sub-networks:
        # Branch 1 (1x1 with coordinate attention gating)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # Branch 2 (3x3 spatial context capture)
        x2 = self.conv3x3(group_x)

        # 5. Cross-spatial learning interaction
        x11 = self.softmax(self.agp(x1).reshape(b * g, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * g, gc, -1)

        x21 = self.softmax(self.agp(x2).reshape(b * g, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * g, gc, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * g, 1, h, w)

        # 6. Reweight input features and reshape back to (b, c, h, w) with residual identity skip connection
        att = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return x + att
