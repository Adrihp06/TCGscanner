from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.annotations import iter_dataset_images, load_annotation
from riftbound_scanner.region_detector import RegionDetector


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX region detector against local annotations.")
    parser.add_argument("--model", default="models/riftbound_regions.onnx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", default="runs/detect/riftbound_regions_eval.json")
    args = parser.parse_args()

    detector = RegionDetector(args.model)
    if not detector.available:
        raise SystemExit(f"Detector unavailable: {args.model}")

    images = [image for image in iter_dataset_images() if load_annotation(image.image_id).get("boxes")]
    if args.limit:
        images = images[: args.limit]

    summary = {"images": len(images), "labels": {}, "items": []}
    for image in images:
        expected = load_annotation(image.image_id).get("boxes", [])
        detected = [item.as_dict() for item in detector.detect(image.path, confidence_threshold=args.conf)]
        item = {"image_id": image.image_id, "path": image.relative_path, "matches": []}
        for truth in expected:
            same_label = [box for box in detected if box["label"] == truth["label"]]
            best = max((iou(truth, box) for box in same_label), default=0.0)
            label_summary = summary["labels"].setdefault(truth["label"], {"count": 0, "iou_sum": 0.0, "hit_50": 0})
            label_summary["count"] += 1
            label_summary["iou_sum"] += best
            label_summary["hit_50"] += 1 if best >= 0.5 else 0
            item["matches"].append({"label": truth["label"], "best_iou": best})
        summary["items"].append(item)

    for values in summary["labels"].values():
        count = values["count"] or 1
        values["mean_iou"] = values["iou_sum"] / count
        values["recall_50"] = values["hit_50"] / count

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "labels": summary["labels"]}, indent=2))


if __name__ == "__main__":
    main()
