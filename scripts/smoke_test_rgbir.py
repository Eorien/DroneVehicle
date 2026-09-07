#!/usr/bin/env python3
"""Run a focused RGB-IR dataset, augmentation, dataloader, and OBB forward smoke test."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from dronevehicle import RGBIRDataset, initialize_dual_stream, initialize_ir_channel
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import RGBIRSplit
from ultralytics.nn.tasks import OBBModel, load_checkpoint

DEFAULT_MODEL = Path(
    "third_party/ultralytics/ultralytics/cfg/models/11/yolo11-obb.yaml"
)
DEFAULT_WEIGHTS = Path("weights/yolo11n-obb.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="configs/data/dronevehicle_rgbir.yaml",
        help="RGB-IR dataset YAML",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="YOLO OBB model YAML"
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Official YOLO11n-OBB checkpoint",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dataset_kwargs(data: dict, hyp: object, batch_size: int) -> dict:
    return {
        "imgsz": 640,
        "batch_size": batch_size,
        "hyp": hyp,
        "cache": False,
        "stride": 32,
        "single_cls": False,
        "classes": None,
        "data": data,
        "task": "obb",
    }


def check_raw_channels(data: dict) -> None:
    dataset = RGBIRDataset(
        img_path=data["val"],
        augment=False,
        rect=False,
        pad=0.5,
        **dataset_kwargs(data, copy(DEFAULT_CFG), 4),
    )
    raw, _, _ = dataset.load_image(0, rect_mode=False)
    record = dataset.records_by_rgb[dataset.im_files[0]]
    rgb_bgr = cv2.imread(dataset.im_files[0], cv2.IMREAD_COLOR)
    infrared = cv2.imread(record["ir"], cv2.IMREAD_GRAYSCALE)
    expected_rgb = cv2.resize(
        cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )
    expected_ir = cv2.resize(infrared, (640, 640), interpolation=cv2.INTER_LINEAR)
    assert raw.shape == (640, 640, 4)
    assert np.array_equal(raw[..., :3], expected_rgb), (
        "RGB content or channel order changed"
    )
    assert np.array_equal(raw[..., 3], expected_ir), "IR content changed"
    sample = dataset[0]
    assert tuple(sample["img"].shape) == (4, 640, 640)
    assert sample["img"].dtype == torch.uint8
    assert sample["bboxes"].shape[1] == 5
    print("raw RGB + IR content: OK")


def check_weight_initialization(
    data: dict, model_yaml: str, weights_path: str
) -> OBBModel:
    if not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"Official pretrained weights not found: {weights_path}"
        )
    source, _ = load_checkpoint(weights_path, device="cpu")
    source_weight = source.state_dict()["model.0.conv.weight"].float().clone()
    assert source.task == "obb" and source_weight.shape[1] == 3
    target = OBBModel(model_yaml, ch=4, nc=len(data["names"]), verbose=False)
    if isinstance(target.model[0], RGBIRSplit):
        transferred = initialize_dual_stream(target, source)
        target_state = target.state_dict()
        assert transferred > 0
        assert torch.equal(target_state["model.2.conv.weight"], source_weight)
        assert torch.equal(
            target_state["model.14.conv.weight"],
            source_weight.mean(dim=1, keepdim=True),
        )
        print("dual RGB/IR backbone transfer and equal-gate initialization: OK")
    else:
        target.load(source, verbose=False)
        assert initialize_ir_channel(target, source)
        target_weight = target.state_dict()["model.0.conv.weight"]
        assert torch.equal(target_weight[:, :3], source_weight)
        assert torch.equal(
            target_weight[:, 3:4], source_weight.mean(dim=1, keepdim=True)
        )
        print("pretrained RGB transfer and IR mean initialization: OK")
    print(f"official checkpoint: {weights_path}")
    return target


def check_augmented_forward(
    data: dict, args: argparse.Namespace, model: OBBModel
) -> None:
    hyp = copy(DEFAULT_CFG)
    hyp.hsv_h = hyp.hsv_s = hyp.hsv_v = 0.0
    hyp.bgr = 0.0
    dataset = RGBIRDataset(
        img_path=data["train"],
        augment=True,
        rect=False,
        pad=0.0,
        fraction=args.train_samples,
        **dataset_kwargs(data, hyp, args.batch),
    )
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        device=args.device,
        pin_memory=True,
    )
    try:
        batch = next(iter(loader))
        assert batch["img"].ndim == 4 and batch["img"].shape[1] == 4
        assert batch["img"].dtype == torch.uint8
        assert batch["bboxes"].shape[1] == 5

        model = model.eval().to(args.device)
        images = batch["img"].to(args.device, non_blocking=True).float() / 255.0
        with torch.inference_mode():
            output = model(images)
        if isinstance(model.model[0], RGBIRSplit):
            assert model.model[2].conv.weight.shape[1] == 3
            assert model.model[14].conv.weight.shape[1] == 1
            input_description = "RGB first conv: 3 channels; IR first conv: 1 channel"
        else:
            assert model.model[0].conv.weight.shape[1] == 4
            input_description = (
                f"first convolution: {tuple(model.model[0].conv.weight.shape)}"
            )
        assert isinstance(output, tuple) and output[0].shape[0] == images.shape[0]
        print(f"augmented batch: {tuple(batch['img'].shape)}")
        print(input_description)
        print(f"OBB prediction: {tuple(output[0].shape)}")
        print("augmented GPU forward: OK")
    finally:
        loader.close()


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.workers < 0 or args.train_samples < args.batch:
        raise ValueError("Require batch > 0, workers >= 0, and train-samples >= batch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the default smoke test but is not available"
        )
    torch.manual_seed(0)
    data = check_det_dataset(args.data, autodownload=False)
    if data["channels"] != 4:
        raise ValueError(f"Expected four data channels, got {data['channels']}")
    check_raw_channels(data)
    model = check_weight_initialization(data, args.model, args.weights)
    check_augmented_forward(data, args, model)


if __name__ == "__main__":
    main()
