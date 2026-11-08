"""Efficient Multi-Scale Attention (EMA) Module with Cross-Spatial Learning.

Dual-compatible implementation supporting:
1. ICASSP 2023 formulation with registered pooling operators (Fold 0: PaddyYolo_Enhanced.pt)
2. Mean-context formulation without registered submodules (Fold 2 and alternative checkpoints)
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

        Adaptively routes to the architecture variant corresponding to the checkpoint:
        - If pool_h is present (Fold 0): applies the ICASSP coordinate pooling + cross-attention with residual skip.
        - If pool_h is absent: applies the mean-context grouped query formulation.
        """
        if hasattr(self, "pool_h") and getattr(self, "pool_h") is not None:
            # 1. Feature grouping: reshape channels into batch dimension
            b, c, h, w = x.size()
            g = getattr(self, "groups", 8)
            gc = c // g
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

            # 6. Reweight input features with residual identity skip connection
            att = (group_x * weights.sigmoid()).reshape(b, c, h, w)
            return x + att
        else:
            # Fallback for checkpoints serialized without registered pooling submodules
            batch, channels, height, width = x.shape
            groups = getattr(self, "groups", 8)
            grouped = x.reshape(batch * groups, channels // groups, height, width)

            height_context = grouped.mean(dim=3, keepdim=True)
            width_context = grouped.mean(dim=2, keepdim=True).transpose(2, 3)

            context = self.conv1x1(torch.cat((height_context, width_context), dim=2))
            height_gate, width_gate = context.split((height, width), dim=2)

            gated = self.gn(grouped * height_gate.sigmoid() * width_gate.transpose(2, 3).sigmoid())
            spatial = self.conv3x3(grouped)

            gated_query = gated.mean((2, 3)).unsqueeze(1).softmax(dim=-1)
            spatial_query = spatial.mean((2, 3)).unsqueeze(1).softmax(dim=-1)

            weights = (gated_query @ spatial.flatten(2) + spatial_query @ gated.flatten(2))
            weights = weights.reshape(batch * groups, 1, height, width).sigmoid()

            return (grouped * weights).reshape(batch, channels, height, width)
