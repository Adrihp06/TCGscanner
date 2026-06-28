from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_detector import DEFAULT_REPO_ID, download_detector

ROOT = Path(__file__).resolve().parent.parent


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a reproducible local Riftbound scanner setup.")
    parser.add_argument("--detector-repo", default=DEFAULT_REPO_ID)
    parser.add_argument("--force-model", action="store_true")
    parser.add_argument("--sets", default="", help="Comma-separated official set codes. Defaults to every set.")
    parser.add_argument("--limit", type=int, default=0, help="Limit catalog import/index size for smoke tests.")
    parser.add_argument("--catalog-output", default="data/official_cards.json")
    parser.add_argument("--db", default="data/vector_db")
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--skip-index", action="store_true", help="Download model and catalog, but skip LanceDB indexing.")
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    download_detector(repo_id=args.detector_repo, force=args.force_model)

    import_cmd = [
        sys.executable,
        "scripts/import_official_cards.py",
        "--output",
        args.catalog_output,
    ]
    if args.sets:
        import_cmd.extend(["--sets", args.sets])
    if args.limit:
        import_cmd.extend(["--limit", str(args.limit)])
    if args.overwrite_images:
        import_cmd.append("--overwrite-images")
    run_step(import_cmd)

    if not args.skip_index:
        index_cmd = [
            sys.executable,
            "scripts/build_vector_index.py",
            "--catalog",
            args.catalog_output,
            "--db",
            args.db,
            "--table",
            args.table,
        ]
        if args.limit:
            index_cmd.extend(["--limit", str(args.limit)])
        run_step(index_cmd)

    print("\nSetup complete.")
    print("- Detector: models/riftbound_regions.onnx")
    print(f"- Catalog: {args.catalog_output}")
    print(f"- Vector DB: {args.db}/{args.table}")
    print("\nRun the scanner:")
    print("RIFTBOUND_PRICE_PROVIDER=pricecharting uv run python -m riftbound_scanner.server --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
