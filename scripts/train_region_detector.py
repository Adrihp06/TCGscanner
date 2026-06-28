from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n region detector for Riftbound cards.")
    parser.add_argument("--data", default="dataset/yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="8")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="riftbound_regions")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}. Run scripts/prepare_yolo_dataset.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch) if str(args.batch).isdigit() else args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "degrees": 8,
        "translate": 0.08,
        "scale": 0.25,
        "shear": 2,
        "perspective": 0.0005,
        "fliplr": 0.0,
        "flipud": 0.0,
    }
    if args.device:
        train_kwargs["device"] = args.device
    result = model.train(**train_kwargs)
    print(result)


if __name__ == "__main__":
    main()
