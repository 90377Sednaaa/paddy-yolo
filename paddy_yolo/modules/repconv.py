"""RepConv module with structural re-parameterization.

Based on RepVGG (Ding et al., CVPR 2021) and adapted for Paddy-YOLO.
During training, RepConv features a multi-branch topology (3x3 conv+BN, 1x1 conv+BN,
and identity+BN when input and output shapes match).
During inference/deployment, switch_to_deploy() collapses all branches into an equivalent
single 3x3 convolution layer with zero additional inference latency.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, autopad


class RepConv(nn.Module):
    """RepConv layer with structural re-parameterization.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        k: Kernel size (default: 3).
        s: Stride (default: 1).
        p: Padding (default: None, auto-calculated to k // 2).
        g: Groups (default: 1).
        act: Activation function or boolean flag (default: True -> SiLU).
        deploy: Whether to initialize directly in deploy mode (default: False).
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: Union[bool, nn.Module] = True,
        deploy: bool = False,
    ):
        super().__init__()
        assert k == 3, f"RepConv only supports kernel size 3, got k={k}"
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = autopad(k, p)
        self.g = g
        self.deploy = deploy
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.conv = nn.Conv2d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=k,
                stride=s,
                padding=self.p,
                groups=g,
                bias=True,
            )
        else:
            # Identity branch: active only if in_channels == out_channels and stride == 1
            self.bn = nn.BatchNorm2d(num_features=c1) if (c1 == c2 and s == 1) else None
            # Dense 3x3 conv + BN branch
            self.conv1 = Conv(c1, c2, k=k, s=s, p=self.p, g=g, act=False)
            # 1x1 conv + BN branch (padded to match output spatial resolution)
            p_1x1 = self.p - k // 2
            self.conv2 = Conv(c1, c2, k=1, s=s, p=p_1x1, g=g, act=False)

    @property
    def rbr_dense(self) -> Optional[Conv]:
        """RepVGG alias for the 3x3 conv+BN branch."""
        return getattr(self, "conv1", None)

    @property
    def rbr_1x1(self) -> Optional[Conv]:
        """RepVGG alias for the 1x1 conv+BN branch."""
        return getattr(self, "conv2", None)

    @property
    def rbr_identity(self) -> Optional[nn.BatchNorm2d]:
        """RepVGG alias for the identity BN branch."""
        return getattr(self, "bn", None)

    @property
    def bn_identity(self) -> Optional[nn.BatchNorm2d]:
        """Alias for the identity BN branch."""
        return getattr(self, "bn", None)

    @property
    def rbr_reparam(self) -> Optional[nn.Conv2d]:
        """RepVGG alias for the fused single Conv2d layer."""
        return getattr(self, "conv", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        In deploy mode, runs single fused convolution.
        In training mode, runs multi-branch topology and sums branch outputs.
        """
        if self.deploy:
            return self.act(self.conv(x))

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for fused convolution (Ultralytics compatibility)."""
        return self.act(self.conv(x))

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1: Optional[torch.Tensor]) -> Union[torch.Tensor, int]:
        """Pad a 1x1 kernel tensor into a 3x3 kernel tensor."""
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Optional[Union[Conv, nn.BatchNorm2d]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse a branch (Conv+BN or identity BN) into equivalent kernel and bias tensors.

        Math:
            W_fused = W * (gamma / sqrt(var + eps))
            b_fused = beta - mean * (gamma / sqrt(var + eps)) + b_conv
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            bias = branch.conv.bias if branch.conv.bias is not None else 0
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            # Identity branch: create 3x3 identity convolution kernel
            input_dim = self.c1 // self.g
            kernel = torch.zeros(
                (self.c1, input_dim, 3, 3),
                device=branch.weight.device,
                dtype=branch.weight.dtype,
            )
            for i in range(self.c1):
                kernel[i, i % input_dim, 1, 1] = 1.0
            bias = 0
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            raise TypeError(f"Unsupported branch type for fusion: {type(branch)}")

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        w_fused = kernel * t
        b_fused = beta + (bias - running_mean) * gamma / std
        return w_fused, b_fused

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute equivalent 3x3 kernel and bias by summing all branches."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        w_total = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        b_total = bias3x3 + bias1x1 + biasid
        return w_total, b_total

    def switch_to_deploy(self) -> None:
        """Fuse multi-branch topology into a single equivalent Conv2d layer."""
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=self.k,
            stride=self.s,
            padding=self.p,
            groups=self.g,
            bias=True,
        )
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        for p in self.parameters():
            p.detach_()

        # Remove training branch modules to free memory and prevent redundant computation
        if hasattr(self, "conv1"):
            self.__delattr__("conv1")
        if hasattr(self, "conv2"):
            self.__delattr__("conv2")
        if hasattr(self, "bn"):
            self.__delattr__("bn")

        self.deploy = True

    def fuse_convs(self) -> None:
        """Alias for switch_to_deploy (Ultralytics model.fuse compatibility)."""
        self.switch_to_deploy()
