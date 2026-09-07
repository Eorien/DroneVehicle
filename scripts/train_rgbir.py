#!/usr/bin/env python3
"""Train an RGB-IR YOLO11-OBB model with explicit formal or smoke settings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dronevehicle import RGBIROBBTrainer
from ultralytics.utils import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train/rgbir_baseline.yaml"
DEFAULT_AMP_WEIGHTS = PROJECT_ROOT / "weights/yolo26n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--weights", type=Path, help="Override the pretrained checkpoint path"
    )
    parser.add_argument(
        "--device", help="Override the training device, for example 0 or 0,1"
    )
    parser.add_argument("--name", help="Override the run name")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 32 train images, 1 epoch, imgsz 64, batch 4, and workers 2",
    )
    parser.add_argument(
        "--smoke-amp",
        action="store_true",
        help="Enable FP16 AMP during --smoke (off by default for the fastest smoke test)",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    if args.smoke_amp and not args.smoke:
        raise ValueError("--smoke-amp requires --smoke")
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    os.chdir(PROJECT_ROOT)
    overrides = YAML.load(config_path)
    if args.weights:
        overrides["pretrained"] = str(args.weights)
    if args.device:
        overrides["device"] = args.device
    if args.name:
        overrides["name"] = args.name

    model_path = resolve_project_path(overrides["model"])
    data_path = resolve_project_path(overrides["data"])
    weights_path = resolve_project_path(overrides["pretrained"])
    for description, path in (
        ("model YAML", model_path),
        ("data YAML", data_path),
        ("pretrained weights", weights_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    overrides.update(
        model=str(model_path), data=str(data_path), pretrained=str(weights_path)
    )

    if args.smoke:
        overrides.update(
            epochs=1,
            batch=4,
            imgsz=64,
            workers=2,
            fraction=32,
            amp=args.smoke_amp,
            plots=False,
            save=False,
            val=False,
            project="/tmp/dronevehicle-smoke",
            name=args.name or "rgbir-baseline-smoke",
            exist_ok=True,
        )
        print("SMOKE MODE: reduced settings are not formal experiment settings")
    else:
        print(f"FORMAL MODE: {config_path}")

    if overrides.get("amp") is True and not DEFAULT_AMP_WEIGHTS.is_file():
        raise FileNotFoundError(f"AMP check weights not found: {DEFAULT_AMP_WEIGHTS}")

    trainer = RGBIROBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
