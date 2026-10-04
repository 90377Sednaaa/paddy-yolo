"""RGGConv, RGGBottleneck, and RGG modules for Paddy-YOLO.

Based on Section 2.2.1 and Figure 6 of the Paddy-YOLO paper (2025).
In Paddy-YOLO, RGG replaces standard C2f blocks in the YOLOv8 backbone.
It combines:
- Structural re-parameterization via RepConv (3x3 receptive field with zero extra inference latency).
- Dual GhostConvs with 5x5 depthwise convolutions (expanding receptive fields for small pest appendages).
- Multi-scale channel concatenation and fusion via 1x1 projection.
- C2f-style split-and-merge gradient flow topology across sequential RGGBottleneck blocks.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

from paddy_yolo.modules.ghostconv import GhostConv
from paddy_yolo.modules.repconv import RepConv


class RGGConv(nn.Module):
    """RGG Convolution layer combining RepConv and dual GhostConvs.

    Figure 6(b) of Paddy-YOLO:
    - Input tensor passes in parallel through:
      1) RepConv(c1, c2)
      2) GhostConv(c1, c2, k=3, dw_k=5)
      3) GhostConv(c1, c2, k=3, dw_k=5)
    - The outputs are concatenated (3 * c2 channels).
    - A 1x1 Conv projects the concatenated feature maps to c2 channels.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        act: Activation function or boolean flag (default: True -> SiLU).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        self.rep = RepConv(c1, c2, k=3, s=1, act=act)
        self.ghost1 = GhostConv(c1, c2, k=3, s=1, dw_k=5, act=act)
        self.ghost2 = GhostConv(c1, c2, k=3, s=1, dw_k=5, act=act)
        self.conv = Conv(3 * c2, c2, k=1, s=1, act=act)

        # Aliases for convenience / conventional access
        self.cv1 = self.rep
        self.cv2 = self.ghost1
        self.cv3 = self.ghost2
        self.fuse = self.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: parallel RepConv and GhostConvs, concatenated and fused.

        Args:
            x: Input tensor of shape (B, c1, H, W).

        Returns:
            Output tensor of shape (B, c2, H, W).
        """
        y1 = self.rep(x)
        y2 = self.ghost1(x)
        y3 = self.ghost2(x)
        return self.conv(torch.cat((y1, y2, y3), dim=1))

    def switch_to_deploy(self) -> None:
        """Switch internal RepConv branch to deploy mode."""
        if hasattr(self.rep, "switch_to_deploy"):
            self.rep.switch_to_deploy()


class RGGBottleneck(nn.Module):
    """RGGBottleneck residual module.

    Figure 6(c) of Paddy-YOLO:
    - Conv(c1, c2, 1) -> RGGConv(c2, c2).
    - Shortcut connection: x + path(x) when shortcut=True and c1 == c2, else path(x).

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        shortcut: Whether to enable residual shortcut connection (default: True).
        g: Groups for convolution (default: 1).
        k: Kernel size tuple (default: (1, 3)).
        e: Expansion ratio (default: 0.5).
        act: Activation function or boolean flag (default: True).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Union[int, Tuple[int, int]] = (1, 3),
        e: float = 0.5,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.shortcut = shortcut
        self.g = g
        self.e = e
        self.add = shortcut and c1 == c2

        k1 = k[0] if isinstance(k, (tuple, list)) else k
        self.cv1 = Conv(c1, c2, k=k1, s=1, act=act)
        self.cv2 = RGGConv(c2, c2, act=act)
        self.rgg_conv = self.cv2  # alias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual shortcut + RGGConv(Conv(x)).

        Args:
            x: Input tensor of shape (B, c1, H, W).

        Returns:
            Output tensor of shape (B, c2, H, W).
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    def switch_to_deploy(self) -> None:
        """Switch internal RGGConv to deploy mode."""
        if hasattr(self.cv2, "switch_to_deploy"):
            self.cv2.switch_to_deploy()


class RGG(nn.Module):
    """RGG feature extraction module replacing C2f in YOLOv8 backbone.

    Figure 6(d) of Paddy-YOLO:
    Follows CSP (Cross Stage Partial) split-and-merge topology:
    - cv1 projects c1 -> 2 * c (where c = int(c2 * e)).
    - The output is split into two halves of c channels each.
    - One half bypasses directly; the second half is sequentially processed by
      n RGGBottleneck blocks.
    - All intermediate bottleneck outputs and the bypass are concatenated.
    - cv2 projects the concatenated (2 + n) * c channels back to c2.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        n: Number of RGGBottleneck blocks (default: 1).
        shortcut: Whether to use shortcut connection in bottlenecks (default: True).
        g: Groups for convolutions (default: 1).
        e: Hidden channel expansion ratio (default: 0.5).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RGGBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(1, 3), e=e)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RGG module."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk() (export friendly)."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def switch_to_deploy(self) -> None:
        """Switch all internal RepConv layers to deploy mode."""
        for m in self.m:
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
