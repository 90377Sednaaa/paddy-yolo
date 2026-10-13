"""Efficient Multi-Scale Attention (EMA), not an exponential moving average.

Algorithm reference: https://github.com/YOLOonMe/EMA-attention-module
Grouped height/width attention and a 3x3 spatial branch preserve tensor shape.
"""
import torch
from torch import nn


class EMA(nn.Module):
    """Apply cross-spatial attention independently to groups of channels."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        if groups < 1 or channels % groups:
            raise ValueError("EMA channels must be divisible by a positive group count")
        self.groups = groups
        grouped_channels = channels // groups
        self.gn = nn.GroupNorm(grouped_channels, grouped_channels)
        self.conv1x1 = nn.Conv2d(grouped_channels, grouped_channels, 1)
        self.conv3x3 = nn.Conv2d(grouped_channels, grouped_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch * self.groups, channels // self.groups, height, width)
        height_context = grouped.mean(dim=3, keepdim=True)
        width_context = grouped.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.conv1x1(torch.cat((height_context, width_context), dim=2))
        height_gate, width_gate = context.split((height, width), dim=2)
        gated = self.gn(grouped * height_gate.sigmoid() * width_gate.transpose(2, 3).sigmoid())
        spatial = self.conv3x3(grouped)
        gated_query = gated.mean((2, 3)).unsqueeze(1).softmax(dim=-1)
        spatial_query = spatial.mean((2, 3)).unsqueeze(1).softmax(dim=-1)
        weights = (gated_query @ spatial.flatten(2) + spatial_query @ gated.flatten(2))
        weights = weights.reshape(batch * self.groups, 1, height, width).sigmoid()
        return (grouped * weights).reshape(batch, channels, height, width)
