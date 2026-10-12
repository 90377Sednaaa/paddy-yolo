"""GhostConv module with 5x5 Depthwise Convolution.

Based on GhostNet (Han et al., CVPR 2020) and customized for Paddy-YOLO (Section 2.2.1).
In Paddy-YOLO, GhostConv serves as the 2nd and 3rd convolution layers of RGGConv.
To enlarge the receptive field and capture richer spatial features, the standard
primary convolution uses kernel size 3x3 (k=3), while the cheap transformation uses
a 5x5 depthwise convolution (dw_k=5) with groups equal to ceil(c_out / 2).
"""

import math
from typing import Union

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class GhostConv(nn.Module):
    """Ghost Convolution with 5x5 Depthwise Convolution.

    Splits feature generation into a primary standard convolution and a cheap
    5x5 depthwise convolution to expand receptive fields with low computational cost.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        k: Kernel size of primary convolution (default: 3).
        s: Stride of primary convolution (default: 1).
        dw_k: Kernel size of cheap depthwise convolution (default: 5).
        act: Activation function or boolean flag (default: True -> SiLU).
        g: Groups for primary convolution (default: 1).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        dw_k: int = 5,
        act: Union[bool, nn.Module] = True,
        g: int = 1,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.dw_k = dw_k

        if c2 < 1:
            raise ValueError("GhostConv requires at least one output channel")
        # Round up and trim so odd channel counts still use true depthwise Conv.
        c_ = math.ceil(c2 / 2)
        c_dw = c_
        dw_groups = c_

        # Primary convolution: standard Conv (c1 -> ceil(c2/2), kernel k, stride s)
        self.cv1 = Conv(c1, c_, k=k, s=s, p=k // 2, g=g, act=act)

        # Cheap transformation: DWConv (c_ -> c_, kernel dw_k, stride 1)
        self.cv2 = Conv(c_, c_dw, k=dw_k, s=1, p=dw_k // 2, g=dw_groups, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation: concatenate primary features and cheap DWConv features.

        Args:
            x: Input tensor of shape (B, C1, H, W).

        Returns:
            Concatenated output tensor of shape (B, C2, H/s, W/s).
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)[:, :self.c2]
