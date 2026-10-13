"""Learned point-sampling upsampler (DySample, ICCV 2023).

Adapted from https://github.com/tiny-smart/dysample (MIT, Wenze Liu, 2023).
See third_party/dysample-LICENSE.txt. No custom CUDA extension is required.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DySample(nn.Module):
    """Upsample with learned grouped offsets; LP/PL variants follow the reference."""

    def __init__(self, channels: int, scale: int = 2, style: str = "lp",
                 groups: int = 4, dyscope: bool = False):
        super().__init__()
        if style not in ("lp", "pl") or scale < 1:
            raise ValueError("DySample requires style lp/pl and a positive integer scale")
        if groups < 1 or channels % groups:
            raise ValueError("DySample channels must be divisible by a positive group count")
        if style == "pl" and channels % (scale * scale):
            raise ValueError("PL requires channels divisible by scale squared")
        self.scale, self.style, self.groups = scale, style, groups
        offset_channels = channels // (scale * scale) if style == "pl" else channels
        offset_outputs = 2 * groups if style == "pl" else 2 * groups * scale * scale
        self.offset = nn.Conv2d(offset_channels, offset_outputs, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        if dyscope:
            self.scope = nn.Conv2d(offset_channels, offset_outputs, 1, bias=False)
            nn.init.zeros_(self.scope.weight)
        positions = (torch.arange(scale, dtype=torch.float32) - (scale - 1) / 2) / scale
        initial = torch.stack(torch.meshgrid(positions, positions, indexing="ij"))
        initial = initial.transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1)
        self.register_buffer("init_pos", initial)

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        offset = offset.reshape(batch, 2, -1, height, width)
        xs = torch.arange(width, device=x.device, dtype=offset.dtype) + 0.5
        ys = torch.arange(height, device=x.device, dtype=offset.dtype) + 0.5
        coordinates = torch.stack(torch.meshgrid(xs, ys, indexing="ij"))
        coordinates = coordinates.transpose(1, 2)[None, :, None]
        normalizer = offset.new_tensor((width, height)).reshape(1, 2, 1, 1, 1)
        grid = 2 * (coordinates + offset) / normalizer - 1
        grid = F.pixel_shuffle(grid.reshape(batch, -1, height, width), self.scale)
        grid = grid.reshape(batch, 2, self.groups, height * self.scale, width * self.scale)
        grid = grid.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        result = F.grid_sample(x.reshape(batch * self.groups, channels // self.groups, height, width),
                               grid, mode="bilinear", padding_mode="border", align_corners=False)
        return result.reshape(batch, channels, height * self.scale, width * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_input = F.pixel_shuffle(x, self.scale) if self.style == "pl" else x
        offset = self.offset(offset_input)
        offset = offset * self.scope(offset_input).sigmoid() * 0.5 if hasattr(self, "scope") else offset * 0.25
        if self.style == "pl":
            offset = F.pixel_unshuffle(offset, self.scale)
        return self.sample(x, offset + self.init_pos)
