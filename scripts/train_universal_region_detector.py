from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.universal_yolo import prepare_universal_yolo_dataset


EXPERIMENTS = ["localization_only", "hybrid"]
DEFAULT_AUGMENTATION = {
    "degrees": 12,
    "translate": 0.10,
    "scale": 0.35,
    "shear": 3,
    "perspective": 0.001,
    "hsv_s": 0.25,
    "hsv_v": 0.25,
    "fliplr": 0.0,
    "flipud": 0.0,
}


def metric_value(metrics: Any, name: str, default: float = 0.0) -> float:
    value = getattr(metrics.box, name, default)
    try:
        return float(value)
    except TypeError:
        return default


def metrics_summary(metrics: Any) -> dict[str, float]:
    return {
        "map50": metric_value(metrics, "map50"),
        "map50_95": metric_value(metrics, "map"),
        "precision": metric_value(metrics, "mp"),
        "recall": metric_value(metrics, "mr"),
    }


def collect_sample_images(run_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    patterns = ["val_batch*_pred.jpg", "val_batch*_labels.jpg", "results.png", "confusion_matrix.png"]
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern))[:8]:
            destination = output_dir / path.name
            shutil.copy2(path, destination)
            copied.append(str(destination))
    return copied


def train_experiment(args: argparse.Namespace, experiment: str) -> dict[str, Any]:
    from ultralytics import YOLO

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    conversion = prepare_universal_yolo_dataset(
        source_dir=Path(args.source),
        output_root=Path(args.yolo_output_root),
        experiment=experiment,
        seed=args.seed,
        limit=args.limit,
    )
    model = YOLO(args.model)
    run_name = f"tcg_{experiment}"
    train_kwargs: dict[str, Any] = {
        "data": conversion["data_yaml"],
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch) if str(args.batch).isdigit() else args.batch,
        "project": args.project,
        "name": run_name,
        "exist_ok": True,
        "patience": args.patience,
        "seed": args.seed,
        **DEFAULT_AUGMENTATION,
    }
    if args.device:
        train_kwargs["device"] = args.device

    train_result = model.train(**train_kwargs)
    run_dir = Path(train_result.save_dir)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"Expected trained weights missing: {weights}")

    trained = YOLO(str(weights))
    val_metrics = trained.val(data=str(conversion["data_yaml"]), split="val", imgsz=args.imgsz, batch=train_kwargs["batch"])
    test_metrics = trained.val(data=str(conversion["data_yaml"]), split="test", imgsz=args.imgsz, batch=train_kwargs["batch"])
    samples = collect_sample_images(run_dir, Path(args.samples_dir) / experiment)
    result = {
        "experiment": experiment,
        "run_dir": str(run_dir),
        "weights": str(weights),
        "conversion": {key: conversion[key] for key in conversion if key != "items"},
        "train_args": train_kwargs,
        "augmentation": DEFAULT_AUGMENTATION,
        "val_metrics": metrics_summary(val_metrics),
        "test_metrics": metrics_summary(test_metrics),
        "sample_artifacts": samples,
    }
    (run_dir / "training_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def export_onnx(weights: Path, output: Path, imgsz: int) -> str:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    return str(output)


def write_audit(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(results, key=lambda item: item["test_metrics"]["map50"])
    usable = best["test_metrics"]["map50"] >= args.usable_map50
    export_path = ""
    if usable:
        export_path = export_onnx(Path(best["weights"]), Path(args.export_output), args.imgsz)

    audit = {
        "objective": "train universal single-class TCG card region detector",
        "thresholds": {"usable_map50": args.usable_map50},
        "decision": {
            "usable": usable,
            "best_experiment": best["experiment"],
            "best_test_map50": best["test_metrics"]["map50"],
            "exported_model": export_path,
        },
        "experiments": results,
        "notes": [
            "corners and polygon annotations measure localization labels.",
            "full_image annotations are isolated card images and are reported separately in conversion audits.",
            "Hugging Face tokens are not recorded in this audit.",
        ],
    }
    output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and audit universal TCG card detector experiments.")
    parser.add_argument("--source", default="dataset/universal_tcg_detection")
    parser.add_argument("--yolo-output-root", default="dataset/yolo_universal")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="8")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--samples-dir", default="reports/region_detector_samples")
    parser.add_argument("--audit-output", default="reports/region_detector_training_audit.json")
    parser.add_argument("--export-output", default="models/riftbound_regions.onnx")
    parser.add_argument("--usable-map50", type=float, default=0.75)
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [train_experiment(args, experiment) for experiment in args.experiments]
    audit = write_audit(args, results)
    print(json.dumps(audit["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
