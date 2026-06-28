from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.official_gallery import DEFAULT_OFFICIAL_PATH, import_official_cards


def parse_set_codes(value: str) -> list[str] | None:
    codes = [item.strip().upper() for item in value.split(",") if item.strip()]
    return codes or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official Riftbound cards and images.")
    parser.add_argument("--output", default=str(DEFAULT_OFFICIAL_PATH))
    parser.add_argument(
        "--sets",
        default="",
        help="Comma-separated official set codes. Defaults to every set exposed by the gallery.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    payload = import_official_cards(
        output_path=Path(args.output),
        set_codes=parse_set_codes(args.sets),
        limit=args.limit,
        download_images=not args.metadata_only,
        overwrite_images=args.overwrite_images,
    )
    sets = ", ".join(f"{item['set_code']}:{item['card_count']}" for item in payload["sets"])
    print(f"Imported {len(payload['cards'])} official cards into {args.output} ({sets})")


if __name__ == "__main__":
    main()
