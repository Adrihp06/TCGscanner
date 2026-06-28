from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.universal_yolo import prepare_universal_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert universal TCG annotations to YOLO datasets.")
    parser.add_argument("--source", default="dataset/universal_tcg_detection")
    parser.add_argument("--output-root", default="dataset/yolo_universal")
    parser.add_argument("--experiment", choices=["localization_only", "hybrid"], required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    audit = prepare_universal_yolo_dataset(
        source_dir=Path(args.source),
        output_root=Path(args.output_root),
        experiment=args.experiment,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps({key: audit[key] for key in ["experiment", "data_yaml", "counts", "total"]}, indent=2))


if __name__ == "__main__":
    main()
