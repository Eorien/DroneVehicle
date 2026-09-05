"""Paired RGB-IR dataset support for frozen DroneVehicle OBB manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_hash
from ultralytics.utils.patches import imread


class RGBIRDataset(YOLODataset):
    """Load frozen RGB and IR pairs as a single four-channel YOLO OBB image."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data = kwargs.get("data") or {}
        if kwargs.get("task", "detect") != "obb":
            raise ValueError("RGBIRDataset currently supports only task='obb'")
        if data.get("channels") != 4:
            raise ValueError("RGBIRDataset requires data['channels'] == 4")
        cache = kwargs.get("cache")
        if isinstance(cache, str) and cache.lower() == "disk":
            raise ValueError(
                "RGBIRDataset does not write disk caches beside the read-only source images"
            )
        self.manifest_path: Path | None = None
        self.records_by_rgb: dict[str, dict[str, str]] = {}
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read one frozen JSONL manifest and return its RGB image paths."""
        if isinstance(img_path, list):
            if len(img_path) != 1:
                raise ValueError("RGBIRDataset accepts exactly one manifest per split")
            img_path = img_path[0]
        manifest_path = Path(img_path).resolve()
        if manifest_path.suffix.lower() != ".jsonl" or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen RGB-IR manifest not found: {manifest_path}"
            )

        frozen_root = manifest_path.parent.parent
        data_root = frozen_root.parent
        records = []
        seen_rgb = set()
        with manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, 1):
                try:
                    record = json.loads(line)
                    rgb_path = (data_root / record["rgb_image"]).resolve()
                    ir_path = (data_root / record["ir_image"]).resolve()
                    label_path = (frozen_root / record["label"]).resolve()
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"Invalid record in {manifest_path}:{line_number}"
                    ) from error
                if (
                    not rgb_path.is_file()
                    or not ir_path.is_file()
                    or not label_path.is_file()
                ):
                    raise FileNotFoundError(
                        f"Missing RGB, IR, or label file in {manifest_path}:{line_number}"
                    )
                rgb_key = str(rgb_path)
                if rgb_key in seen_rgb:
                    raise ValueError(
                        f"Duplicate RGB path in {manifest_path}:{line_number}: {rgb_path}"
                    )
                seen_rgb.add(rgb_key)
                records.append((rgb_key, str(ir_path), str(label_path)))

        if not records:
            raise ValueError(f"Manifest contains no records: {manifest_path}")
        count = (
            self.fraction
            if isinstance(self.fraction, int)
            else max(1, round(len(records) * self.fraction))
        )
        records = records[:count] if count < len(records) else records
        self.manifest_path = manifest_path
        self.records_by_rgb = {
            rgb: {"ir": ir, "label": label} for rgb, ir, label in records
        }
        return [rgb for rgb, _, _ in records]

    def get_label_files(self) -> list[str]:
        """Return frozen RGB-reference labels in the same order as RGB images."""
        self.label_files = [
            self.records_by_rgb[str(Path(rgb_path).resolve())]["label"]
            for rgb_path in self.im_files
        ]
        return self.label_files

    def get_cache_hash(self) -> str:
        """Include RGB, IR, labels, and the manifest in label-cache invalidation."""
        ir_files = [
            self.records_by_rgb[str(Path(rgb_path).resolve())]["ir"]
            for rgb_path in self.im_files
        ]
        manifest = [str(self.manifest_path)] if self.manifest_path else []
        return get_hash(self.label_files + self.im_files + ir_files + manifest)

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load RGB and paired grayscale IR, then return one RGB-IR HWC array."""
        cached = self.ims[i]
        if cached is not None:
            return cached, self.im_hw0[i], self.im_hw[i]

        rgb_file = self.im_files[i]
        record = self.records_by_rgb[str(Path(rgb_file).resolve())]
        rgb = imread(rgb_file, flags=cv2.IMREAD_COLOR)
        infrared = imread(record["ir"], flags=cv2.IMREAD_GRAYSCALE)
        if rgb is None or infrared is None:
            raise FileNotFoundError(
                f"Unable to decode RGB-IR pair: {rgb_file}, {record['ir']}"
            )
        if rgb.shape[:2] != infrared.shape[:2]:
            raise ValueError(
                f"RGB-IR shape mismatch: {rgb_file} {rgb.shape[:2]} != {record['ir']} {infrared.shape[:2]}"
            )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if infrared.ndim == 2:
            infrared = infrared[..., None]
        if infrared.ndim != 3 or infrared.shape[2] != 1:
            raise ValueError(
                f"Expected one IR channel, got shape {infrared.shape} from {record['ir']}"
            )
        image = np.concatenate((rgb, infrared), axis=2)
        h0, w0 = image.shape[:2]
        if rect_mode:
            if resize_short:
                ratio = self.imgsz / min(h0, w0)
                if ratio != 1:
                    width, height = (
                        (math.ceil(w0 * ratio), self.imgsz)
                        if h0 < w0
                        else (self.imgsz, math.ceil(h0 * ratio))
                    )
                    image = cv2.resize(
                        image, (width, height), interpolation=cv2.INTER_LINEAR
                    )
            else:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    width = min(math.ceil(w0 * ratio), self.imgsz)
                    height = min(math.ceil(h0 * ratio), self.imgsz)
                    image = cv2.resize(
                        image, (width, height), interpolation=cv2.INTER_LINEAR
                    )
        elif not (h0 == w0 == self.imgsz):
            image = cv2.resize(
                image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
            )

        if self.augment and self.cache != "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                image,
                (h0, w0),
                image.shape[:2],
            )
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                old_index = self.buffer.pop(0)
                if self.cache != "ram":
                    (
                        self.ims[old_index],
                        self.im_hw0[old_index],
                        self.im_hw[old_index],
                    ) = None, None, None
        return image, (h0, w0), image.shape[:2]
