from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.official_gallery import DEFAULT_UNLEASHED_PATH, import_official_cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official Riftbound Unleashed cards and images.")
    parser.add_argument("--output", default=str(DEFAULT_UNLEASHED_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    payload = import_official_cards(
        output_path=Path(args.output),
        set_codes=["UNL"],
        limit=args.limit,
        download_images=not args.metadata_only,
        overwrite_images=args.overwrite_images,
    )
    print(f"Imported {len(payload['cards'])} official Unleashed cards into {args.output}")


if __name__ == "__main__":
    main()
