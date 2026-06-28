from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "images"
ANNOTATION_DIR = ROOT / "annotations"
DATASET_DIR = ROOT / "dataset" / "yolo"
LABELS = ["card"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetImage:
    image_id: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class AnnotationIssue:
    image_id: str
    message: str


def image_id(relative_path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", relative_path)


def annotation_path(image_id_value: str) -> Path:
    return ANNOTATION_DIR / f"{image_id_value}.json"


def iter_dataset_images(image_root: Path = IMAGE_DIR) -> list[DatasetImage]:
    images: list[DatasetImage] = []
    if not image_root.exists():
        return images
    for path in sorted(image_root.glob("**/*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(ROOT).as_posix()
        images.append(DatasetImage(image_id=image_id(rel), path=path, relative_path=rel))
    return images


def load_annotation(image_id_value: str) -> dict[str, object]:
    path = annotation_path(image_id_value)
    if not path.exists():
        return {"image_id": image_id_value, "labels": LABELS, "boxes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_annotations(require_all_labels: bool = True) -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    try:
        from PIL import Image
    except ImportError:
        return [AnnotationIssue("environment", "Pillow is required to validate image sizes.")]

    for image in iter_dataset_images():
        payload = load_annotation(image.image_id)
        boxes = payload.get("boxes", [])
        if not boxes:
            continue
        with Image.open(image.path) as img:
            width, height = img.size
        seen_labels: set[str] = set()
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                issues.append(AnnotationIssue(image.image_id, f"box {index} is not an object"))
                continue
            label = str(box.get("label", ""))
            seen_labels.add(label)
            if label not in LABEL_TO_ID:
                issues.append(AnnotationIssue(image.image_id, f"box {index} has invalid label {label!r}"))
            x = float(box.get("x", 0))
            y = float(box.get("y", 0))
            box_width = float(box.get("width", 0))
            box_height = float(box.get("height", 0))
            if box_width <= 0 or box_height <= 0:
                issues.append(AnnotationIssue(image.image_id, f"box {index} has non-positive size"))
            if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                issues.append(AnnotationIssue(image.image_id, f"box {index} is outside image bounds"))
        if require_all_labels:
            missing = set(LABELS) - seen_labels
            if missing:
                issues.append(AnnotationIssue(image.image_id, f"missing labels: {', '.join(sorted(missing))}"))
    return issues


def annotated_images() -> list[DatasetImage]:
    result = []
    for image in iter_dataset_images():
        if load_annotation(image.image_id).get("boxes"):
            result.append(image)
    return result


def prepare_yolo_dataset(
    output_dir: Path = DATASET_DIR,
    val_ratio: float = 0.2,
    seed: int = 7,
    require_all_labels: bool = True,
) -> dict[str, object]:
    issues = validate_annotations(require_all_labels=require_all_labels)
    if issues:
        return {
            "ok": False,
            "issues": [issue.__dict__ for issue in issues],
            "message": "Fix annotation issues before preparing YOLO dataset.",
        }

    images = annotated_images()
    if not images:
        return {
            "ok": False,
            "issues": [{"image_id": "dataset", "message": "No annotated images found."}],
            "message": "Annotate at least a few images before preparing YOLO dataset.",
        }

    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_ids = {image.image_id for image in shuffled[:val_count]}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to prepare YOLO labels.") from exc

    counts = {"train": 0, "val": 0}
    for image in shuffled:
        split = "val" if image.image_id in val_ids else "train"
        with Image.open(image.path) as img:
            width, height = img.size
        destination_image = output_dir / "images" / split / image.path.name
        shutil.copy2(image.path, destination_image)
        labels = []
        for box in load_annotation(image.image_id).get("boxes", []):
            label = str(box["label"])
            x = float(box["x"])
            y = float(box["y"])
            box_width = float(box["width"])
            box_height = float(box["height"])
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            labels.append(
                f"{LABEL_TO_ID[label]} {center_x:.8f} {center_y:.8f} "
                f"{box_width / width:.8f} {box_height / height:.8f}"
            )
        (output_dir / "labels" / split / f"{image.path.stem}.txt").write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )
        counts[split] += 1

    yaml = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml.extend(f"  {index}: {label}" for index, label in enumerate(LABELS))
    (output_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    return {"ok": True, "counts": counts, "data_yaml": str(output_dir / "data.yaml")}


def issues_to_text(issues: Iterable[dict[str, str]], limit: int = 20) -> str:
    lines = []
    for index, issue in enumerate(issues):
        if index >= limit:
            lines.append(f"... {index + 1 - limit} more issue(s)")
            break
        lines.append(f"- {issue['image_id']}: {issue['message']}")
    return "\n".join(lines)
