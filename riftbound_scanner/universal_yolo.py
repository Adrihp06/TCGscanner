from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "dataset" / "universal_tcg_detection"
DEFAULT_OUTPUT = ROOT / "dataset" / "yolo_universal"
LABELS = ["card"]
LOCALIZATION_TYPES = {"corners", "polygon"}
ALL_TYPES = {"corners", "polygon", "full_image"}


@dataclass(frozen=True)
class UniversalRecord:
    id: str
    source: str
    source_tcg: str
    image_path: str
    width: int
    height: int
    annotation_type: str
    corners: list[list[float]]


def bbox_from_points(points: Iterable[Iterable[float]]) -> list[float]:
    coords = [[float(x), float(y)] for x, y in points]
    if not coords:
        raise ValueError("at least one point is required")
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def yolo_line_from_corners(corners: list[list[float]], width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x, y, box_width, box_height = bbox_from_points(corners)
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    values = [center_x, center_y, normalized_width, normalized_height]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"normalized bbox outside image bounds: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def read_records(source_dir: Path = DEFAULT_SOURCE) -> list[UniversalRecord]:
    manifest = source_dir / "annotations.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Universal manifest not found: {manifest}")

    records: list[UniversalRecord] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            obj = row["objects"][0]
            records.append(
                UniversalRecord(
                    id=str(row["id"]),
                    source=str(row["source"]),
                    source_tcg=str(row["source_tcg"]),
                    image_path=str(row["image_path"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    annotation_type=str(obj["annotation_type"]),
                    corners=[[float(x), float(y)] for x, y in obj["corners"]],
                )
            )
    return records


def filter_records(records: Iterable[UniversalRecord], experiment: str) -> list[UniversalRecord]:
    if experiment == "localization_only":
        allowed = LOCALIZATION_TYPES
    elif experiment == "hybrid":
        allowed = ALL_TYPES
    else:
        raise ValueError("experiment must be 'localization_only' or 'hybrid'")
    return [record for record in records if record.annotation_type in allowed]


def split_records(
    records: list[UniversalRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 7,
) -> dict[str, list[UniversalRecord]]:
    if not records:
        raise ValueError("no records to split")
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    if len(shuffled) >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def prepare_universal_yolo_dataset(
    source_dir: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT,
    experiment: str = "localization_only",
    seed: int = 7,
    limit: int = 0,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    experiment_output = output_root.resolve() / experiment
    records = filter_records(read_records(source_dir), experiment)
    if limit:
        records = records[:limit]
    splits = split_records(records, seed=seed)

    if experiment_output.exists():
        shutil.rmtree(experiment_output)
    for split in ["train", "val", "test"]:
        (experiment_output / "images" / split).mkdir(parents=True, exist_ok=True)
        (experiment_output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    audit_items = []
    for split, split_records_value in splits.items():
        counts[split] = len(split_records_value)
        for index, record in enumerate(split_records_value):
            image_source = source_dir / record.image_path
            if not image_source.exists():
                raise FileNotFoundError(f"Missing image for {record.id}: {image_source}")
            safe_stem = record.id.replace("/", "__").replace(" ", "_")
            image_destination = experiment_output / "images" / split / f"{safe_stem}{image_source.suffix.lower()}"
            label_destination = experiment_output / "labels" / split / f"{safe_stem}.txt"
            shutil.copy2(image_source, image_destination)
            label_destination.write_text(
                yolo_line_from_corners(record.corners, record.width, record.height) + "\n",
                encoding="utf-8",
            )
            audit_items.append(
                {
                    "id": record.id,
                    "split": split,
                    "source": record.source,
                    "source_tcg": record.source_tcg,
                    "annotation_type": record.annotation_type,
                    "image": image_destination.relative_to(experiment_output).as_posix(),
                    "label": label_destination.relative_to(experiment_output).as_posix(),
                    "order": index,
                }
            )

    data_yaml = experiment_output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {experiment_output}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: card",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    by_annotation = Counter(item["annotation_type"] for item in audit_items)
    by_tcg = Counter(item["source_tcg"] for item in audit_items)
    by_source = Counter(item["source"] for item in audit_items)
    audit = {
        "experiment": experiment,
        "seed": seed,
        "source_dir": str(source_dir),
        "output_dir": str(experiment_output),
        "data_yaml": str(data_yaml),
        "counts": counts,
        "total": len(audit_items),
        "counts_by_annotation_type": dict(sorted(by_annotation.items())),
        "counts_by_tcg": dict(sorted(by_tcg.items())),
        "counts_by_source": dict(sorted(by_source.items())),
        "items": audit_items,
    }
    (experiment_output / "conversion_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit
