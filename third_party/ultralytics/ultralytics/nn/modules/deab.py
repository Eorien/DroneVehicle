# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detail-enhanced convolution and attention modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("DEAB", "DetailEnhancedConv")


class DetailEnhancedConv(nn.Module):
    """Combine vanilla and directional-difference kernels into one 3x3 convolution.

    The five trainable branches follow the DEConv formulation from DEA-Net:
    vanilla, central-difference, angular-difference, horizontal-difference, and
    vertical-difference convolution. Their kernels are combined before applying
    convolution, so no branch feature maps need to be retained.
    """

    _ANGULAR_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)

    def __init__(self, channels: int) -> None:
        """Initialize five same-channel trainable kernel branches."""
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        self.central = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.angular = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.horizontal = nn.Conv1d(channels, channels, 3, padding=1, bias=True)
        self.vertical = nn.Conv1d(channels, channels, 3, padding=1, bias=True)
        self.vanilla = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    @staticmethod
    def _central_kernel(weight: torch.Tensor) -> torch.Tensor:
        """Convert a vanilla 3x3 kernel into a central-difference kernel."""
        center_mask = weight.new_zeros((1, 1, 3, 3))
        center_mask[..., 1, 1] = 1
        return weight - weight.sum(dim=(-2, -1), keepdim=True) * center_mask

    @classmethod
    def _angular_kernel(cls, weight: torch.Tensor) -> torch.Tensor:
        """Convert a vanilla 3x3 kernel into an angular-difference kernel."""
        flat = weight.flatten(2)
        permutation = torch.as_tensor(cls._ANGULAR_PERMUTATION, device=weight.device)
        return (flat - flat.index_select(2, permutation)).view_as(weight)

    @staticmethod
    def _horizontal_kernel(weight: torch.Tensor) -> torch.Tensor:
        """Expand a 1D kernel into opposing left/right columns."""
        zeros = torch.zeros_like(weight)
        return torch.stack((weight, zeros, -weight), dim=-1)

    @staticmethod
    def _vertical_kernel(weight: torch.Tensor) -> torch.Tensor:
        """Expand a 1D kernel into opposing top/bottom rows."""
        zeros = torch.zeros_like(weight)
        return torch.stack((weight, zeros, -weight), dim=-2)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the combined 3x3 kernel and bias used by the forward pass."""
        kernel = (
            self._central_kernel(self.central.weight)
            + self._angular_kernel(self.angular.weight)
            + self._horizontal_kernel(self.horizontal.weight)
            + self._vertical_kernel(self.vertical.weight)
            + self.vanilla.weight
        )
        bias = self.central.bias + self.angular.bias + self.horizontal.bias + self.vertical.bias + self.vanilla.bias
        return kernel, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the equivalent detail-enhanced convolution."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"DetailEnhancedConv expects NCHW input with {self.channels} channels, got {tuple(x.shape)}"
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        return F.conv2d(x, kernel, bias, stride=1, padding=1)


class _SpatialAttention(nn.Module):
    """Produce a shared spatial-attention prior from channel statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return self.conv(torch.cat((mean, maximum), dim=1))


class _ChannelAttention(nn.Module):
    """Produce a channel-attention prior from global average-pooled features."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x))


class _PixelAttention(nn.Module):
    """Generate a channel-specific spatial gate from features and coarse attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            channels * 2,
            channels,
            7,
            padding=3,
            padding_mode="reflect",
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if prior.shape != x.shape:
            prior = prior.expand_as(x)
        interleaved = torch.stack((x, prior), dim=2).flatten(1, 2)
        return self.conv(interleaved).sigmoid()


class DEAB(nn.Module):
    """Detail-Enhanced Attention Block from DEA-Net, adapted as a shape-preserving feature block.

    Reference: Z. Chen, Z. He, and Z.-M. Lu, "DEA-Net: Single Image Dehazing
    Based on Detail-Enhanced Convolution and Content-Guided Attention," IEEE
    Transactions on Image Processing, 2024. https://arxiv.org/abs/2301.04805
    """

    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 8) -> None:
        """Initialize a same-channel DEConv, refinement convolution, and attention gate."""
        super().__init__()
        if kernel_size != 3:
            raise ValueError(f"DEAB currently requires kernel_size=3, got {kernel_size}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        self.channels = channels
        self.detail = DetailEnhancedConv(channels)
        self.activation = nn.ReLU(inplace=True)
        self.refine = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)
        self.spatial_attention = _SpatialAttention()
        self.channel_attention = _ChannelAttention(channels, reduction)
        self.pixel_attention = _PixelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance details, estimate coarse attention, and apply a residual pixel gate."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"DEAB expects NCHW input with {self.channels} channels, got {tuple(x.shape)}")
        if min(x.shape[-2:]) <= 3:
            raise ValueError(
                f"DEAB reflect-padded 7x7 attention requires spatial dimensions > 3, got {tuple(x.shape[-2:])}"
            )
        features = self.activation(self.detail(x)) + x
        features = self.refine(features)
        prior = self.channel_attention(features) + self.spatial_attention(features)
        gate = self.pixel_attention(features, prior)
        return x + features * gate
