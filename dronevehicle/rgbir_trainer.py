"""Project-level OBB trainer for four-channel RGB-IR early fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .rgbir_dataset import RGBIRDataset

FIRST_CONV_KEY = "model.0.conv.weight"


def _source_model(weights: Any) -> nn.Module | None:
    if isinstance(weights, dict):
        return weights.get("ema") or weights.get("model")
    return weights if isinstance(weights, nn.Module) else None


def initialize_ir_channel(model: OBBModel, weights: Any) -> bool:
    """Initialize a new fourth input channel from the mean of loaded RGB weights.

    Returns True only when a three-channel source was expanded to four channels.
    Existing four-channel checkpoints are preserved unchanged.
    """
    source = _source_model(weights)
    if source is None:
        return False
    source_weight = source.float().state_dict().get(FIRST_CONV_KEY)
    target_weight = model.state_dict().get(FIRST_CONV_KEY)
    if source_weight is None or target_weight is None:
        raise ValueError(f"First convolution {FIRST_CONV_KEY!r} was not found")
    if target_weight.shape[1] != 4:
        raise ValueError(
            f"RGB-IR model must have four input channels, got {target_weight.shape[1]}"
        )
    if source_weight.shape[1] == 4:
        return False
    if (
        source_weight.shape[1] != 3
        or source_weight.shape[0] != target_weight.shape[0]
        or source_weight.shape[2:] != target_weight.shape[2:]
    ):
        raise ValueError(
            f"Cannot expand first convolution from {tuple(source_weight.shape)} to {tuple(target_weight.shape)}"
        )

    source_weight = source_weight.to(
        device=target_weight.device, dtype=target_weight.dtype
    )
    if not torch.equal(target_weight[:, :3], source_weight):
        raise RuntimeError(
            "Pretrained RGB channels were not transferred exactly before IR initialization"
        )
    with torch.no_grad():
        target_weight[:, 3:4].copy_(source_weight.mean(dim=1, keepdim=True))
    LOGGER.info(
        "Initialized input channel 4 (IR) as the mean of pretrained RGB convolution weights"
    )
    return True


class RGBIROBBTrainer(OBBTrainer):
    """Use frozen paired RGB-IR manifests with the standard Ultralytics OBB training loop."""

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> RGBIRDataset:
        """Build the paired four-channel dataset for training or validation."""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return RGBIRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="obb",
            classes=self.args.classes,
            data=self.data,
            fraction=get_split_fraction(self.args.fraction, mode),
        )

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> OBBModel:
        """Build a four-channel OBB model and explicitly initialize its IR input weights."""
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        initialized = initialize_ir_channel(model, weights)
        if weights is not None and not initialized and RANK in {-1, 0}:
            source = _source_model(weights)
            source_channels = (
                source.state_dict()[FIRST_CONV_KEY].shape[1]
                if source is not None
                else "unknown"
            )
            LOGGER.info(
                f"Preserved existing first-convolution weights with {source_channels} input channels"
            )
        return model
