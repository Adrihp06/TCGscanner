from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained Riftbound region detector to ONNX.")
    parser.add_argument("--weights", default="runs/detect/riftbound_regions/weights/best.pt")
    parser.add_argument("--output", default="models/riftbound_regions.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Train the detector first.")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    print(f"Exported ONNX: {output}")


if __name__ == "__main__":
    main()
