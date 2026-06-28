from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.annotations import issues_to_text, prepare_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Riftbound annotations to YOLO format.")
    parser.add_argument("--output", default="dataset/yolo")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()

    result = prepare_yolo_dataset(
        output_dir=Path(args.output),
        val_ratio=args.val_ratio,
        seed=args.seed,
        require_all_labels=not args.allow_missing_labels,
    )
    if not result["ok"]:
        print(result["message"])
        print(issues_to_text(result["issues"]))
        raise SystemExit(1)
    print(f"YOLO dataset ready: {result['data_yaml']}")
    print(f"Images: train={result['counts']['train']} val={result['counts']['val']}")


if __name__ == "__main__":
    main()
