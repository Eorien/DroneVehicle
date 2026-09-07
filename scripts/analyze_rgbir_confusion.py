#!/usr/bin/env python3
"""Analyze RGB-IR OBB class confusions on val without modifying the frozen dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dronevehicle import RGBIRDataset
from ultralytics.data.build import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import batch_probiou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "configs/data/dronevehicle_rgbir.yaml"
DEFAULT_MODELS = (
    f"baseline={PROJECT_ROOT / 'outputs/formal_experiments/rgbir_baseline/train/weights/best.pt'}",
    f"spd={PROJECT_ROOT / 'outputs/formal_experiments/rgbir_spd/train/weights/best.pt'}",
)
DEFAULT_RARE_CLASSES = ("van", "Freight_car")


class ErrorAnalysisValidator(OBBValidator):
    """Collect class-agnostic OBB matches before the standard validator updates metrics."""

    def __init__(
        self, *args: Any, analysis_conf: float, analysis_iou: float, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.analysis_conf = analysis_conf
        self.analysis_iou = analysis_iou
        self.error_records: list[dict[str, Any]] = []

    @staticmethod
    def _greedy_matches(iou: torch.Tensor, threshold: float) -> np.ndarray:
        """Match GT and predictions by IoU only, following Ultralytics confusion-matrix logic."""
        indices = torch.where(iou > threshold)
        if indices[0].numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)
        matches = (
            torch.cat(
                (torch.stack(indices, 1), iou[indices[0], indices[1]][:, None]), 1
            )
            .cpu()
            .numpy()
        )
        if len(matches) > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches

    def _collect_errors(
        self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]
    ) -> None:
        """Store one record per GT object plus unmatched false-positive detections."""
        for image_index, pred in enumerate(preds):
            prepared = self._prepare_batch(image_index, batch)
            keep = pred["conf"] > self.analysis_conf
            filtered = {key: value[keep] for key, value in pred.items()}
            gt_cls = prepared["cls"].int()
            gt_boxes = prepared["bboxes"]
            pred_cls = filtered["cls"].int()
            pred_boxes = filtered["bboxes"]

            if len(gt_boxes) and len(pred_boxes):
                matches = self._greedy_matches(
                    batch_probiou(gt_boxes, pred_boxes), self.analysis_iou
                )
            else:
                matches = np.zeros((0, 3), dtype=np.float32)
            gt_to_pred = {int(gt): (int(det), float(iou)) for gt, det, iou in matches}
            matched_predictions = {int(det) for _, det, _ in matches}

            gt_native = ops.scale_boxes(
                prepared["imgsz"],
                gt_boxes.clone(),
                prepared["ori_shape"],
                ratio_pad=prepared["ratio_pad"],
                xywh=True,
            )
            pred_native = (
                self.scale_preds(filtered, prepared)["bboxes"]
                if len(pred_boxes)
                else pred_boxes
            )
            image_path = str(Path(prepared["im_file"]).resolve())

            for gt_index, true_id in enumerate(gt_cls.tolist()):
                if gt_index in gt_to_pred:
                    pred_index, iou = gt_to_pred[gt_index]
                    predicted_id = int(pred_cls[pred_index])
                    outcome = "correct" if predicted_id == true_id else "misclassified"
                    confidence = float(filtered["conf"][pred_index])
                    predicted_box = pred_native[pred_index].detach().cpu().tolist()
                else:
                    predicted_id = self.nc
                    outcome = "missed"
                    confidence = None
                    iou = None
                    predicted_box = None
                self.error_records.append(
                    {
                        "image": image_path,
                        "outcome": outcome,
                        "true_id": true_id,
                        "true_class": self.names[true_id],
                        "predicted_id": predicted_id,
                        "predicted_class": "background"
                        if predicted_id == self.nc
                        else self.names[predicted_id],
                        "confidence": confidence,
                        "iou": iou,
                        "gt_xywhr": gt_native[gt_index].detach().cpu().tolist(),
                        "pred_xywhr": predicted_box,
                    }
                )

            for pred_index, predicted_id in enumerate(pred_cls.tolist()):
                if pred_index in matched_predictions:
                    continue
                self.error_records.append(
                    {
                        "image": image_path,
                        "outcome": "false_positive",
                        "true_id": self.nc,
                        "true_class": "background",
                        "predicted_id": predicted_id,
                        "predicted_class": self.names[predicted_id],
                        "confidence": float(filtered["conf"][pred_index]),
                        "iou": None,
                        "gt_xywhr": None,
                        "pred_xywhr": pred_native[pred_index].detach().cpu().tolist(),
                    }
                )

    def update_metrics(
        self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]
    ) -> None:
        """Collect error records, then preserve the standard OBB validation behavior."""
        self._collect_errors(preds, batch)
        super().update_metrics(preds, batch)


def parse_args() -> argparse.Namespace:
    """Parse model, split, matching, and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="NAME=best.pt; repeat for multiple models",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Prediction confidence for confusion analysis",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Class-agnostic OBB IoU matching threshold",
    )
    parser.add_argument("--rare-class", action="append", default=[])
    parser.add_argument("--gallery-size", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/error_analysis/rgbir_val"
    )
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    """Parse and validate repeated NAME=PATH model specifications."""
    values = values or list(DEFAULT_MODELS)
    models = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH for --model, got {value!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        models.append((name, path))
    if len({name for name, _ in models}) != len(models):
        raise ValueError("Model names must be unique")
    return models


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rgbir_dataset(
    data: dict[str, Any], args: argparse.Namespace, validator: OBBValidator
) -> RGBIRDataset:
    """Build the unchanged four-channel dataset for standalone validation."""
    return RGBIRDataset(
        img_path=data[args.split],
        imgsz=640,
        batch_size=args.batch,
        augment=False,
        hyp=validator.args,
        rect=True,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix=f"{args.split}: ",
        task="obb",
        classes=None,
        data=data,
        fraction=1.0,
    )


def write_matrix_csv(
    path: Path, matrix: np.ndarray, labels: list[str], normalize: bool
) -> None:
    """Write predicted-row, true-column confusion counts or normalized rates."""
    values = (
        matrix / (matrix.sum(axis=0, keepdims=True) + 1e-9) if normalize else matrix
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, values):
            writer.writerow(
                [label, *[f"{value:.6f}" if normalize else int(value) for value in row]]
            )


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write per-object matching records as a flat CSV."""
    fieldnames = (
        "image",
        "outcome",
        "true_id",
        "true_class",
        "predicted_id",
        "predicted_class",
        "confidence",
        "iou",
        "gt_xywhr",
        "pred_xywhr",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "gt_xywhr": json.dumps(record["gt_xywhr"]),
                    "pred_xywhr": json.dumps(record["pred_xywhr"]),
                }
            )


def rare_summary(
    records: list[dict[str, Any]], rare_classes: tuple[str, ...]
) -> dict[str, Any]:
    """Summarize correct, missed, and misclassified outcomes for each rare true class."""
    summary = {}
    for class_name in rare_classes:
        class_records = [
            record for record in records if record["true_class"] == class_name
        ]
        predictions = Counter(record["predicted_class"] for record in class_records)
        outcomes = Counter(record["outcome"] for record in class_records)
        total = len(class_records)
        summary[class_name] = {
            "total_ground_truth": total,
            "outcomes": dict(outcomes),
            "predicted_as_counts": dict(predictions),
            "predicted_as_rates": {
                name: count / total for name, count in predictions.items()
            },
        }
    return summary


def rbox_polygon(box: list[float]) -> np.ndarray:
    """Convert one native-space xywhr box to four integer polygon vertices."""
    tensor = torch.tensor(box, dtype=torch.float32).view(1, 5)
    return ops.xywhr2xyxyxyxy(tensor).view(4, 2).round().int().numpy()


def annotated_crop(
    record: dict[str, Any], ir_path: Path, card_size: int = 320
) -> np.ndarray:
    """Create an RGB/IR side-by-side crop with GT in green and prediction in red."""
    rgb = cv2.imread(record["image"], cv2.IMREAD_COLOR)
    infrared = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or infrared is None:
        raise FileNotFoundError(record["image"])
    infrared_bgr = cv2.cvtColor(infrared, cv2.COLOR_GRAY2BGR)
    gt_polygon = rbox_polygon(record["gt_xywhr"])
    pred_polygon = rbox_polygon(record["pred_xywhr"])
    points = np.concatenate((gt_polygon, pred_polygon), axis=0)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    radius = max(70, int(max(x_max - x_min, y_max - y_min) * 2.5))
    left, right = max(0, center_x - radius), min(rgb.shape[1], center_x + radius)
    top, bottom = max(0, center_y - radius), min(rgb.shape[0], center_y + radius)

    panels = []
    for image in (rgb, infrared_bgr):
        cv2.polylines(image, [gt_polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.polylines(image, [pred_polygon], True, (0, 0, 255), 3, cv2.LINE_AA)
        crop = image[top:bottom, left:right]
        panels.append(
            cv2.resize(crop, (card_size, card_size), interpolation=cv2.INTER_LINEAR)
        )
    card = np.concatenate(panels, axis=1)
    title = (
        f"{Path(record['image']).stem}  GT:{record['true_class']}  "
        f"Pred:{record['predicted_class']}  conf:{record['confidence']:.2f}  IoU:{record['iou']:.2f}"
    )
    banner = np.full((40, card.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((banner, card), axis=0)


def write_gallery(
    path: Path,
    records: list[dict[str, Any]],
    ir_by_rgb: dict[str, Path],
    true_class: str,
    predicted_class: str,
    limit: int,
) -> int:
    """Write a two-column contact sheet for one directed class confusion."""
    selected = [
        record
        for record in records
        if record["outcome"] == "misclassified"
        and record["true_class"] == true_class
        and record["predicted_class"] == predicted_class
    ]
    selected.sort(
        key=lambda record: (record["confidence"], record["iou"]), reverse=True
    )
    unique_records = []
    seen_images = set()
    for record in selected:
        if record["image"] in seen_images:
            continue
        seen_images.add(record["image"])
        unique_records.append(record)
        if len(unique_records) == limit:
            break
    cards = [
        annotated_crop(record, ir_by_rgb[record["image"]]) for record in unique_records
    ]
    if not cards:
        return 0
    if len(cards) % 2:
        cards.append(np.full_like(cards[0], 255))
    rows = [
        np.concatenate(cards[index : index + 2], axis=1)
        for index in range(0, len(cards), 2)
    ]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))
    return len(unique_records)


def analyze_model(
    name: str,
    weights: Path,
    data: dict[str, Any],
    args: argparse.Namespace,
    rare_classes: tuple[str, ...],
) -> None:
    """Run validation and export raw confusion evidence for one checkpoint."""
    save_dir = args.output.resolve() / name
    if save_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {save_dir}")
    validator_args = {
        "task": "obb",
        "mode": "val",
        "model": str(weights),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": 640,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "rect": True,
        "cache": False,
        "fraction": 1.0,
        "plots": True,
        "verbose": True,
        "visualize": False,
        "save_json": False,
        "save_txt": False,
        "conf": None,  # NMS becomes 0.01 for OBB, while confusion_matrix_conf remains 0.25
        "iou": 0.7,
        "seed": 0,
        "deterministic": True,
    }
    validator = ErrorAnalysisValidator(
        save_dir=save_dir,
        args=validator_args,
        analysis_conf=args.conf,
        analysis_iou=args.iou,
    )
    dataset = build_rgbir_dataset(data, args, validator)
    validator.dataloader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
        device="cuda" if args.device != "cpu" else "cpu",
    )
    try:
        stats = validator(model=str(weights))
    finally:
        validator.dataloader.close()

    labels = [data["names"][index] for index in range(len(data["names"]))] + [
        "background"
    ]
    matrix = validator.confusion_matrix.matrix
    reconstructed = np.zeros_like(matrix)
    for record in validator.error_records:
        reconstructed[record["predicted_id"], record["true_id"]] += 1
    if not np.array_equal(matrix, reconstructed):
        raise RuntimeError(
            "Per-object records do not reproduce the Ultralytics confusion matrix"
        )
    write_matrix_csv(save_dir / "confusion_counts.csv", matrix, labels, normalize=False)
    write_matrix_csv(
        save_dir / "confusion_normalized.csv", matrix, labels, normalize=True
    )
    write_records(save_dir / "all_matches.csv", validator.error_records)
    rare_records = [
        record
        for record in validator.error_records
        if record["true_class"] in rare_classes
    ]
    write_records(save_dir / "rare_class_matches.csv", rare_records)

    ir_by_rgb = {
        str(Path(rgb_path).resolve()): Path(record["ir"])
        for rgb_path, record in dataset.records_by_rgb.items()
    }
    galleries = {
        "van_to_car": write_gallery(
            save_dir / "van_to_car_gallery.jpg",
            validator.error_records,
            ir_by_rgb,
            "van",
            "car",
            args.gallery_size,
        ),
        "Freight_car_to_truck": write_gallery(
            save_dir / "Freight_car_to_truck_gallery.jpg",
            validator.error_records,
            ir_by_rgb,
            "Freight_car",
            "truck",
            args.gallery_size,
        ),
    }
    metadata = {
        "model_name": name,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data.resolve()),
        "split": args.split,
        "images": len(dataset),
        "analysis_conf": args.conf,
        "analysis_iou": args.iou,
        "matrix_orientation": "rows=predicted, columns=true",
        "stats": {key: float(value) for key, value in stats.items()},
        "rare_classes": rare_summary(validator.error_records, rare_classes),
        "galleries": galleries,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
    }
    (save_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"{name}: analysis saved to {save_dir}")


def main() -> None:
    """Run error analysis for all requested fine-tuned checkpoints."""
    args = parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be in [0, 1]")
    if args.batch <= 0 or args.workers < 0 or args.gallery_size < 0:
        raise ValueError("Require batch > 0, workers >= 0, and gallery-size >= 0")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.data = args.data.resolve()
    data = check_det_dataset(str(args.data), autodownload=False, split=args.split)
    if data["channels"] != 4:
        raise ValueError(
            f"Expected four-channel RGB-IR data, got {data['channels']} channels"
        )
    rare_classes = tuple(args.rare_class or DEFAULT_RARE_CLASSES)
    unknown = sorted(set(rare_classes) - set(data["names"].values()))
    if unknown:
        raise ValueError(f"Unknown rare classes: {unknown}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, weights in parse_models(args.model):
        analyze_model(name, weights, data, args, rare_classes)


if __name__ == "__main__":
    main()
