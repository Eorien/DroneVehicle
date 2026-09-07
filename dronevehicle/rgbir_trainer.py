"""Project-level OBB trainer for four-channel RGB-IR early or dual-stream fusion."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import RGBIRSplit, ShallowCrossModalInteraction
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


def _dual_destination_indices(source_index: int) -> tuple[int, ...]:
    """Map one official YOLO11 layer index to dual RGB/IR branches or the shared head."""
    if 0 <= source_index <= 10:
        return source_index + 2, source_index + 14
    if 11 <= source_index <= 23:
        return (source_index + 17,)
    return ()


def _hybrid_destination_indices(source_index: int) -> tuple[int, ...]:
    """Map one official YOLO11 layer index to the Hybrid Fusion layout."""
    if 0 <= source_index <= 2:
        return source_index + 2, source_index + 6
    if 3 <= source_index <= 10:
        return source_index + 8, source_index + 17
    if 11 <= source_index <= 23:
        return (source_index + 20,)
    return ()


def initialize_dual_stream(model: OBBModel, weights: Any) -> int:
    """Explicitly initialize RGB/IR backbones and the shared head from one RGB checkpoint."""
    source = _source_model(weights)
    if source is None:
        return 0
    if not isinstance(model.model[0], RGBIRSplit):
        raise TypeError(
            "Dual-stream initialization requires RGBIRSplit at model layer 0"
        )
    is_hybrid = any(
        isinstance(module, ShallowCrossModalInteraction) for module in model.model
    )
    destination_indices = (
        _hybrid_destination_indices if is_hybrid else _dual_destination_indices
    )
    ir_first_index = 6 if is_hybrid else 14
    fusion_indices = (28, 29, 30) if is_hybrid else (25, 26, 27)

    source_state = source.float().state_dict()
    target_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    backbone_targets: set[str] = set()

    for source_key, source_tensor in source_state.items():
        match = re.match(r"^model\.(\d+)\.(.+)$", source_key)
        if match is None:
            continue
        source_index = int(match.group(1))
        suffix = match.group(2)
        for destination_index in destination_indices(source_index):
            destination_key = f"model.{destination_index}.{suffix}"
            if destination_key not in target_state:
                skipped.append(f"{source_key} -> {destination_key} (missing target)")
                continue
            target_tensor = target_state[destination_key]
            if (
                source_index == 0
                and destination_index == ir_first_index
                and suffix == "conv.weight"
            ):
                if source_tensor.shape[1] != 3 or target_tensor.shape[1] != 1:
                    raise ValueError(
                        "IR branch first convolution must map from three source channels to one target channel"
                    )
                mapped_tensor = source_tensor.mean(dim=1, keepdim=True)
            elif source_tensor.shape == target_tensor.shape:
                mapped_tensor = source_tensor
            else:
                skipped.append(
                    f"{source_key} -> {destination_key} "
                    f"({tuple(source_tensor.shape)} != {tuple(target_tensor.shape)})"
                )
                continue
            transferred[destination_key] = mapped_tensor.to(
                device=target_tensor.device, dtype=target_tensor.dtype
            )
            if source_index <= 10:
                backbone_targets.add(destination_key)

    expected_backbone_targets = {
        f"model.{destination_index}.{match.group(2)}"
        for source_key in source_state
        if (match := re.match(r"^model\.(\d+)\.(.+)$", source_key))
        and int(match.group(1)) <= 10
        for destination_index in destination_indices(int(match.group(1)))
    }
    missing_backbone = sorted(expected_backbone_targets - backbone_targets)
    if missing_backbone:
        raise RuntimeError(
            f"Dual-stream backbone weights were not fully initialized: {missing_backbone}"
        )

    model.load_state_dict(transferred, strict=False)
    loaded_state = model.state_dict()
    source_first = source_state["model.0.conv.weight"].to(
        loaded_state["model.2.conv.weight"]
    )
    if not torch.equal(loaded_state["model.2.conv.weight"], source_first):
        raise RuntimeError("RGB branch first convolution was not transferred exactly")
    ir_first_key = f"model.{ir_first_index}.conv.weight"
    if not torch.equal(
        loaded_state[ir_first_key], source_first.mean(dim=1, keepdim=True)
    ):
        raise RuntimeError(
            "IR branch first convolution was not initialized as the RGB mean"
        )
    fusion_tensors = [
        tensor
        for key, tensor in loaded_state.items()
        if any(key.startswith(f"model.{index}.gate.") for index in fusion_indices)
    ]
    if not fusion_tensors or any(
        torch.count_nonzero(tensor) for tensor in fusion_tensors
    ):
        raise RuntimeError(
            "Adaptive fusion gates must start at exactly 0.5 via zero logits"
        )
    if is_hybrid:
        projection_tensors = [
            tensor
            for key, tensor in loaded_state.items()
            if key.startswith(("model.9.project_rgb.", "model.9.project_ir."))
        ]
        if not projection_tensors or any(
            torch.count_nonzero(tensor) for tensor in projection_tensors
        ):
            raise RuntimeError(
                "Hybrid residual projections must start at zero perturbation"
            )

    LOGGER.info(
        f"Transferred {len(transferred)}/{len(target_state)} target state items "
        "into both modality backbones and the shared head"
    )
    LOGGER.info(
        "Initialized IR branch input convolution as the mean of pretrained RGB weights"
    )
    LOGGER.info("Initialized P3/P4/P5 adaptive fusion gates to equal RGB/IR weighting")
    if is_hybrid:
        LOGGER.info(
            "Initialized shallow RGB/IR residual projections to zero perturbation"
        )
    if skipped:
        LOGGER.info(
            f"Skipped {len(skipped)} incompatible mapped items (expected task-head differences): "
            + "; ".join(skipped)
        )
    return len(transferred)


def initialize_hybrid_stream(model: OBBModel, weights: Any) -> int:
    """Initialize a Hybrid Fusion model while preserving the pure dual-stream mapping."""
    if not any(
        isinstance(module, ShallowCrossModalInteraction) for module in model.model
    ):
        raise TypeError("Hybrid initialization requires ShallowCrossModalInteraction")
    return initialize_dual_stream(model, weights)


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
        weights: Any = None,
        verbose: bool = True,
    ) -> OBBModel:
        """Build an early- or dual-stream four-channel OBB model with explicit weight transfer."""
        model = self.set_model_names_for_load(
            OBBModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        if isinstance(model.model[0], RGBIRSplit):
            if weights is not None:
                if any(
                    isinstance(module, ShallowCrossModalInteraction)
                    for module in model.model
                ):
                    initialize_hybrid_stream(model, weights)
                else:
                    initialize_dual_stream(model, weights)
            return model

        if weights is not None:
            model.load(weights)
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
