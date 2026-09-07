# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RGB-IR dual-stream feature-fusion modules."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("AdaptiveFeatureFusion", "RGBIRSplit", "ShallowCrossModalInteraction")


class RGBIRSplit(nn.Module):
    """Split a four-channel RGB-IR tensor into RGB and single-channel IR tensors."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(RGB, IR)` without copying channel data."""
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"RGBIRSplit expects NCHW input with 4 channels, got {tuple(x.shape)}")
        return x[:, :3], x[:, 3:4]


class ShallowCrossModalInteraction(nn.Module):
    """Inject one shared shallow representation into two residual modality streams."""

    def __init__(self, channels: int) -> None:
        """Initialize shared 1x1 mixing and zero-output RGB/IR projections."""
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        # Keep all later random initialization identical to the pure dual-stream model.
        with torch.random.fork_rng(devices=[]):
            self.shared = nn.Conv2d(channels * 2, channels, 1, bias=True)
            self.project_rgb = nn.Conv2d(channels, channels, 1, bias=True)
            self.project_ir = nn.Conv2d(channels, channels, 1, bias=True)
            nn.init.zeros_(self.project_rgb.weight)
            nn.init.zeros_(self.project_rgb.bias)
            nn.init.zeros_(self.project_ir.weight)
            nn.init.zeros_(self.project_ir.bias)

    def forward(
        self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-perturbation residual enhancements for RGB and IR features."""
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError("ShallowCrossModalInteraction expects exactly [rgb, infrared]")
        rgb, infrared = features
        if rgb.shape != infrared.shape:
            raise ValueError(
                f"RGB and IR shallow feature shapes must match, got {tuple(rgb.shape)} and {tuple(infrared.shape)}"
            )
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                f"ShallowCrossModalInteraction expects NCHW features with {self.channels} channels, got {tuple(rgb.shape)}"
            )
        shared = self.shared(torch.cat((rgb, infrared), dim=1))
        return rgb + self.project_rgb(shared), infrared + self.project_ir(shared)


class AdaptiveFeatureFusion(nn.Module):
    """Fuse aligned RGB and IR features with a learned channel-spatial gate.

    The gate predicts one value per channel and spatial position:
    `output = gate * rgb + (1 - gate) * infrared`.
    Its 1x1 convolution is initialized to zero, so every gate starts at 0.5.
    """

    def __init__(self, channels: int) -> None:
        """Initialize a shape-preserving gate for two equal-channel feature maps."""
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        # Do not let fusion-only parameters perturb downstream random initialization.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Conv2d(channels * 2, channels, 1, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return an adaptive convex combination of aligned RGB and IR features."""
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError("AdaptiveFeatureFusion expects exactly [rgb, infrared]")
        rgb, infrared = features
        if rgb.shape != infrared.shape:
            raise ValueError(
                f"RGB and IR feature shapes must match, got {tuple(rgb.shape)} and {tuple(infrared.shape)}"
            )
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                f"AdaptiveFeatureFusion expects NCHW features with {self.channels} channels, got {tuple(rgb.shape)}"
            )
        gate = self.gate(torch.cat((rgb, infrared), dim=1)).sigmoid()
        return gate * rgb + (1.0 - gate) * infrared
