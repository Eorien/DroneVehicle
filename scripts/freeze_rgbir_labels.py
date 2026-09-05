#!/usr/bin/env python3
"""Freeze strictly intersected RGB-IR pairs and RGB-reference YOLO OBB labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ("car", "truck", "bus", "van", "Freight_car")


@dataclass(frozen=True)
class OBB:
    class_id: int
    points: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("data/cleaned"),
        help="Cleaned dataset directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/frozen_rgbir"),
        help="Frozen manifests and labels directory",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Minimum rotated IoU for each RGB-IR match",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory",
    )
    return parser.parse_args()


def signed_area(
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> float:
    if len(points) < 3:
        return 0.0
    return (
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2
    )


def polygon_area(
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> float:
    return abs(signed_area(points))


def convex_hull(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return tuple(unique)

    def cross(
        origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def polygon_intersection(
    subject: tuple[tuple[float, float], ...], clip: tuple[tuple[float, float], ...]
) -> list[tuple[float, float]]:
    if len(subject) < 3 or len(clip) < 3:
        return []
    orientation = 1 if signed_area(clip) >= 0 else -1
    output = list(subject)

    def inside(
        point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        return (
            orientation
            * ((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
            >= -1e-9
        )

    def line_intersection(
        start: tuple[float, float],
        end: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> tuple[float, float]:
        subject_dx, subject_dy = end[0] - start[0], end[1] - start[1]
        clip_dx, clip_dy = b[0] - a[0], b[1] - a[1]
        denominator = subject_dx * clip_dy - subject_dy * clip_dx
        if abs(denominator) < 1e-12:
            return end
        distance = (
            (a[0] - start[0]) * clip_dy - (a[1] - start[1]) * clip_dx
        ) / denominator
        return start[0] + distance * subject_dx, start[1] + distance * subject_dy

    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            if inside(end, clip_start, clip_end):
                if not inside(start, clip_start, clip_end):
                    output.append(line_intersection(start, end, clip_start, clip_end))
                output.append(end)
            elif inside(start, clip_start, clip_end):
                output.append(line_intersection(start, end, clip_start, clip_end))
            start = end
    return output


def rotated_iou(first: OBB, second: OBB) -> float:
    first_x = [point[0] for point in first.points]
    first_y = [point[1] for point in first.points]
    second_x = [point[0] for point in second.points]
    second_y = [point[1] for point in second.points]
    if (
        max(first_x) <= min(second_x)
        or max(second_x) <= min(first_x)
        or max(first_y) <= min(second_y)
        or max(second_y) <= min(first_y)
    ):
        return 0.0
    intersection_area = polygon_area(polygon_intersection(first.points, second.points))
    union_area = (
        polygon_area(first.points) + polygon_area(second.points) - intersection_area
    )
    return intersection_area / union_area if union_area > 0 else 0.0


def read_labels(path: Path, width: int, height: int) -> list[OBB]:
    boxes = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(
                f"Expected 9 label fields in {path}:{line_number}, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Non-numeric label in {path}:{line_number}") from error
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"Invalid class ID {class_id} in {path}:{line_number}")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates
        ):
            raise ValueError(f"Invalid normalized coordinate in {path}:{line_number}")
        points = convex_hull(
            [
                (coordinates[index] * width, coordinates[index + 1] * height)
                for index in range(0, 8, 2)
            ]
        )
        if len(points) < 3 or polygon_area(points) <= 0:
            raise ValueError(f"Degenerate OBB in {path}:{line_number}")
        boxes.append(OBB(class_id, points))
    return boxes


def perfect_match(
    rgb_boxes: list[OBB], ir_boxes: list[OBB], threshold: float
) -> list[tuple[int, int, float]]:
    """Find a deterministic full one-to-one matching, or return an empty list."""
    if not rgb_boxes or len(rgb_boxes) != len(ir_boxes):
        return []

    adjacency: list[list[tuple[int, float]]] = []
    for rgb_box in rgb_boxes:
        candidates = []
        for ir_index, ir_box in enumerate(ir_boxes):
            if rgb_box.class_id != ir_box.class_id:
                continue
            iou = rotated_iou(rgb_box, ir_box)
            if iou >= threshold:
                candidates.append((ir_index, iou))
        adjacency.append(sorted(candidates, key=lambda item: (-item[1], item[0])))
    if any(not candidates for candidates in adjacency):
        return []

    ir_to_rgb: dict[int, int] = {}

    def augment(rgb_index: int, seen_ir: set[int]) -> bool:
        for ir_index, _ in adjacency[rgb_index]:
            if ir_index in seen_ir:
                continue
            seen_ir.add(ir_index)
            previous_rgb = ir_to_rgb.get(ir_index)
            if previous_rgb is None or augment(previous_rgb, seen_ir):
                ir_to_rgb[ir_index] = rgb_index
                return True
        return False

    rgb_order = sorted(
        range(len(rgb_boxes)), key=lambda index: (len(adjacency[index]), index)
    )
    if any(not augment(rgb_index, set()) for rgb_index in rgb_order):
        return []

    rgb_to_ir = {rgb_index: ir_index for ir_index, rgb_index in ir_to_rgb.items()}
    return [
        (
            rgb_index,
            rgb_to_ir[rgb_index],
            next(
                iou
                for ir_index, iou in adjacency[rgb_index]
                if ir_index == rgb_to_ir[rgb_index]
            ),
        )
        for rgb_index in range(len(rgb_boxes))
    ]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def value_at(fraction: float) -> float:
        return round(ordered[round((len(ordered) - 1) * fraction)], 6)

    return {
        "p50": value_at(0.5),
        "p90": value_at(0.9),
        "p95": value_at(0.95),
        "p99": value_at(0.99),
    }


def center(box: OBB) -> tuple[float, float]:
    return sum(point[0] for point in box.points) / len(box.points), sum(
        point[1] for point in box.points
    ) / len(box.points)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_split(
    split: str,
    clean_root: Path,
    data_root: Path,
    staging_root: Path,
    threshold: float,
) -> tuple[dict[str, object], list[float], list[float]]:
    source_manifest = clean_root / "manifests" / f"{split}.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing cleaned manifest: {source_manifest}")

    output_manifest = staging_root / "manifests" / f"{split}.jsonl"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_label_dir = staging_root / "labels" / split
    output_label_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    matched_ious: list[float] = []
    center_distances: list[float] = []
    seen_ids = set()
    with (
        source_manifest.open(encoding="utf-8") as source,
        output_manifest.open("w", encoding="utf-8", newline="\n") as output,
    ):
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            sample_id = record["id"]
            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate ID {sample_id!r} in {source_manifest}:{line_number}"
                )
            seen_ids.add(sample_id)
            stats["source_pairs"] += 1

            rgb_image = data_root / record["rgb_image"]
            ir_image = data_root / record["ir_image"]
            if not rgb_image.is_file() or not ir_image.is_file():
                raise FileNotFoundError(f"Missing paired image for {split}/{sample_id}")
            rgb_label = clean_root / record["rgb_label"]
            ir_label = clean_root / record["ir_label"]
            rgb_boxes = read_labels(rgb_label, record["width"], record["height"])
            ir_boxes = read_labels(ir_label, record["width"], record["height"])

            if not rgb_boxes or not ir_boxes:
                stats["dropped_empty_modality"] += 1
                continue
            if len(rgb_boxes) != len(ir_boxes):
                stats["dropped_count_mismatch"] += 1
                continue
            matches = perfect_match(rgb_boxes, ir_boxes, threshold)
            if len(matches) != len(rgb_boxes):
                stats["dropped_incomplete_iou_match"] += 1
                continue

            frozen_label = output_label_dir / f"{sample_id}.txt"
            shutil.copyfile(rgb_label, frozen_label)
            match_records = []
            for rgb_index, ir_index, iou in matches:
                rgb_center, ir_center = (
                    center(rgb_boxes[rgb_index]),
                    center(ir_boxes[ir_index]),
                )
                distance = math.dist(rgb_center, ir_center)
                matched_ious.append(iou)
                center_distances.append(distance)
                class_counts[rgb_boxes[rgb_index].class_id] += 1
                match_records.append(
                    {
                        "rgb_index": rgb_index,
                        "ir_index": ir_index,
                        "rotated_iou": round(iou, 8),
                        "center_distance_px": round(distance, 6),
                    }
                )

            frozen_record = {
                "id": sample_id,
                "width": record["width"],
                "height": record["height"],
                "rgb_image": record["rgb_image"],
                "ir_image": record["ir_image"],
                "label": frozen_label.relative_to(staging_root).as_posix(),
                "rgb_audit_label": rgb_label.relative_to(data_root).as_posix(),
                "ir_audit_label": ir_label.relative_to(data_root).as_posix(),
                "matches": match_records,
            }
            output.write(
                json.dumps(frozen_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stats["frozen_pairs"] += 1
            stats["frozen_objects"] += len(matches)

    result = dict(stats)
    result["class_counts"] = {
        CLASS_NAMES[index]: class_counts[index] for index in range(len(CLASS_NAMES))
    }
    return result, matched_ious, center_distances


def write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_dataset(
    clean_root: Path, staging_root: Path, threshold: float
) -> dict[str, object]:
    data_root = clean_root.parent
    clean_report = clean_root / "report.json"
    if not clean_report.is_file():
        raise FileNotFoundError(f"Missing cleaned report: {clean_report}")

    report: dict[str, object] = {
        "status": "frozen",
        "source_clean_report_sha256": sha256(clean_report),
        "criteria": {
            "same_class": True,
            "one_to_one": True,
            "minimum_rotated_iou": threshold,
            "required_rgb_coverage": 1.0,
            "required_ir_coverage": 1.0,
            "empty_pairs": "drop",
        },
        "supervision": {
            "reference": "RGB",
            "geometry": "cleaned RGB OBB copied unchanged",
            "IR_labels": "quality-control and matching only",
        },
        "splits": {},
    }
    all_ious: list[float] = []
    all_distances: list[float] = []
    total_stats: Counter[str] = Counter()
    total_classes: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_stats, ious, distances = freeze_split(
            split, clean_root, data_root, staging_root, threshold
        )
        report["splits"][split] = split_stats
        all_ious.extend(ious)
        all_distances.extend(distances)
        total_stats.update(
            {key: value for key, value in split_stats.items() if key != "class_counts"}
        )
        total_classes.update(split_stats["class_counts"])

    totals = dict(total_stats)
    totals["class_counts"] = {name: total_classes[name] for name in CLASS_NAMES}
    totals["rotated_iou_quantiles"] = quantiles(all_ious)
    totals["center_distance_px_quantiles"] = quantiles(all_distances)
    report["totals"] = totals
    (staging_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(staging_root)
    return report


def main() -> int:
    args = parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")
    clean_root = args.clean_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == clean_root or output_root in clean_root.parents:
        raise ValueError("Output root must not be the clean root or one of its parents")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}; pass --overwrite to regenerate it"
        )

    staging_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True)
    try:
        report = freeze_dataset(clean_root, staging_root, args.iou_threshold)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    totals = report["totals"]
    print(
        f"Frozen pairs: {totals['frozen_pairs']}; objects: {totals['frozen_objects']}"
    )
    print(f"Output: {output_root}")
    print(f"Report: {output_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
