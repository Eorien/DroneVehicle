#!/usr/bin/env python3
"""Validate and convert DroneVehicle XML annotations to YOLO OBB labels.

The source dataset is treated as read-only. RGB and IR annotations are retained
separately because the two modalities do not contain identical object labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASS_NAMES = ("car", "truck", "bus", "van", "Freight_car")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_ALIASES = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "Freight_car",
    "freight_car": "Freight_car",
    "feright car": "Freight_car",
    "feright_car": "Freight_car",
    "feright": "Freight_car",
}
IGNORED_CLASSES = {"*"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Raw DroneVehicle dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cleaned"),
        help="Generated labels, manifests, and report directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory",
    )
    return parser.parse_args()


def files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate basename in {directory}: {path.stem}")
        result[path.stem] = path
    return result


def require_matching_stems(split: str, groups: dict[str, dict[str, Path]]) -> list[str]:
    names = iter(groups)
    reference_name = next(names)
    reference = set(groups[reference_name])
    errors = []
    for name in names:
        current = set(groups[name])
        missing = sorted(reference - current)
        extra = sorted(current - reference)
        if missing or extra:
            errors.append(
                f"{split}/{name} differs from {reference_name}: "
                f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
            )
    if errors:
        raise ValueError("RGB/IR pairing check failed:\n" + "\n".join(errors))
    return sorted(reference)


def parse_number(element: ET.Element, tag: str) -> float:
    text = element.findtext(tag)
    if text is None:
        raise ValueError(f"missing <{tag}>")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite <{tag}> value")
    return value


def extract_points(obj: ET.Element) -> tuple[list[tuple[float, float]], str]:
    polygon = obj.find("polygon")
    box = obj.find("bndbox")
    if polygon is not None and box is not None:
        raise ValueError("object contains both polygon and bndbox")
    if polygon is not None:
        return [
            (parse_number(polygon, f"x{index}"), parse_number(polygon, f"y{index}"))
            for index in range(1, 5)
        ], "polygon"
    if box is not None:
        xmin = parse_number(box, "xmin")
        ymin = parse_number(box, "ymin")
        xmax = parse_number(box, "xmax")
        ymax = parse_number(box, "ymax")
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], "bndbox"
    raise ValueError("object contains neither polygon nor bndbox")


def cross(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_properly_cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    return cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0


def is_self_crossing(points: list[tuple[float, float]]) -> bool:
    return segments_properly_cross(
        points[0], points[1], points[2], points[3]
    ) or segments_properly_cross(points[1], points[2], points[3], points[0])


def order_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a deterministic perimeter order, starting at the uppermost-leftmost point."""
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    ordered = sorted(
        points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x)
    )
    start = min(
        range(len(ordered)), key=lambda index: (ordered[index][1], ordered[index][0])
    )
    return ordered[start:] + ordered[:start]


def polygon_area(points: list[tuple[float, float]]) -> float:
    return (
        abs(
            sum(
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )
        )
        / 2
    )


def clean_annotation(path: Path) -> tuple[tuple[int, int], list[str], Counter[str]]:
    stats: Counter[str] = Counter(files=1)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Malformed XML {path}: {error}") from error

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {path}")
    width = int(parse_number(size, "width"))
    height = int(parse_number(size, "height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {path}: {(width, height)}")

    output_lines = []
    objects = root.findall("object")
    stats["source_objects"] = len(objects)
    if not objects:
        stats["background_files"] = 1

    for object_index, obj in enumerate(objects):
        raw_name = (obj.findtext("name") or "").strip()
        normalized_name = " ".join(raw_name.lower().split())
        if normalized_name in IGNORED_CLASSES:
            stats["dropped_ignored_class"] += 1
            continue
        class_name = CLASS_ALIASES.get(normalized_name)
        if class_name is None:
            raise ValueError(
                f"Unknown class {raw_name!r} in {path}, object {object_index}"
            )
        if raw_name != class_name:
            stats[f"class_alias:{raw_name}->{class_name}"] += 1

        try:
            points, box_type = extract_points(obj)
        except (TypeError, ValueError):
            stats["dropped_invalid_box"] += 1
            continue
        stats[f"source_box:{box_type}"] += 1
        if box_type == "bndbox":
            stats["converted_bndbox"] += 1

        if is_self_crossing(points):
            stats["reordered_self_crossing"] += 1

        clipped = [
            (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
            for x, y in points
        ]
        if clipped != points:
            stats["clipped_out_of_bounds"] += 1
        ordered = order_points(clipped)
        if polygon_area(ordered) <= 0.0:
            stats["dropped_zero_area"] += 1
            continue

        values = [str(CLASS_IDS[class_name])]
        for x, y in ordered:
            values.extend((f"{x / width:.8f}", f"{y / height:.8f}"))
        output_lines.append(" ".join(values))
        stats["output_objects"] += 1
        stats[f"output_class:{class_name}"] += 1

    if not output_lines:
        stats["output_background_files"] = 1
    return (width, height), output_lines, stats


def merge_stats(target: Counter[str], source: Counter[str]) -> None:
    target.update(source)


def structured_stats(stats: Counter[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    aliases = {}
    source_boxes = {}
    classes = {}
    for key, value in sorted(stats.items()):
        if key.startswith("class_alias:"):
            aliases[key.removeprefix("class_alias:")] = value
        elif key.startswith("source_box:"):
            source_boxes[key.removeprefix("source_box:")] = value
        elif key.startswith("output_class:"):
            classes[key.removeprefix("output_class:")] = value
        else:
            result[key] = value
    result["class_aliases"] = aliases
    result["source_box_types"] = source_boxes
    result["output_classes"] = classes
    return result


def prepare_dataset(data_root: Path, staging_root: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "format": "YOLO OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)",
        "class_names": list(CLASS_NAMES),
        "policies": {
            "modalities": "RGB and IR annotations are retained separately",
            "ignored_classes": sorted(IGNORED_CLASSES),
            "invalid_boxes": "drop and count",
            "axis_aligned_bndbox": "convert to four corner points",
            "out_of_bounds": "clip coordinates to image bounds",
            "point_order": "sort around centroid and start at uppermost-leftmost point",
        },
        "splits": {},
    }
    total_stats: Counter[str] = Counter()

    manifest_dir = staging_root / "manifests"
    manifest_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        split_root = data_root / split
        groups = {
            "rgb_images": files_by_stem(split_root / f"{split}img", IMAGE_SUFFIXES),
            "ir_images": files_by_stem(split_root / f"{split}imgr", IMAGE_SUFFIXES),
            "rgb_labels": files_by_stem(split_root / f"{split}label", {".xml"}),
            "ir_labels": files_by_stem(split_root / f"{split}labelr", {".xml"}),
        }
        stems = require_matching_stems(split, groups)
        modality_stats = {"rgb": Counter(), "ir": Counter()}
        manifest_path = manifest_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
            for stem in stems:
                sizes = {}
                label_paths = {}
                for modality in ("rgb", "ir"):
                    output_label = (
                        staging_root / "labels" / modality / split / f"{stem}.txt"
                    )
                    output_label.parent.mkdir(parents=True, exist_ok=True)
                    size, lines, stats = clean_annotation(
                        groups[f"{modality}_labels"][stem]
                    )
                    sizes[modality] = size
                    label_paths[modality] = output_label
                    output_label.write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
                    merge_stats(modality_stats[modality], stats)
                if sizes["rgb"] != sizes["ir"]:
                    raise ValueError(
                        f"RGB/IR annotation sizes differ for {split}/{stem}: {sizes}"
                    )
                record = {
                    "id": stem,
                    "width": sizes["rgb"][0],
                    "height": sizes["rgb"][1],
                    "rgb_image": groups["rgb_images"][stem]
                    .relative_to(data_root)
                    .as_posix(),
                    "ir_image": groups["ir_images"][stem]
                    .relative_to(data_root)
                    .as_posix(),
                    "rgb_label": label_paths["rgb"]
                    .relative_to(staging_root)
                    .as_posix(),
                    "ir_label": label_paths["ir"].relative_to(staging_root).as_posix(),
                }
                manifest.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

        split_stats = {"pairs": len(stems), "modalities": {}}
        for modality, stats in modality_stats.items():
            split_stats["modalities"][modality] = structured_stats(stats)
            merge_stats(total_stats, stats)
        report["splits"][split] = split_stats

    report["totals"] = structured_stats(total_stats)
    (staging_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == data_root or output_root in data_root.parents:
        raise ValueError("Output root must not be the data root or one of its parents")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}; pass --overwrite to regenerate it"
        )

    staging_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        report = prepare_dataset(data_root, staging_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    totals = report["totals"]
    print(
        f"Prepared {sum(split['pairs'] for split in report['splits'].values())} RGB/IR pairs"
    )
    print(f"Generated labels: {output_root}")
    print(
        f"Retained objects: {totals['output_objects']}; source objects: {totals['source_objects']}"
    )
    print(f"Report: {output_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
