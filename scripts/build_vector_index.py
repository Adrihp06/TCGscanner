from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.official_gallery import DEFAULT_OFFICIAL_PATH, load_official_cards
from riftbound_scanner.vector_store import CardVectorStore
from riftbound_scanner.visual import DEFAULT_MODEL, PREPROCESS_VERSION, CardPreprocessor, ImageEmbedder

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the visual vector index for Riftbound cards.")
    parser.add_argument("--catalog", default=str(DEFAULT_OFFICIAL_PATH))
    parser.add_argument("--db", default=str(ROOT / "data" / "vector_db"))
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cards = load_official_cards(Path(args.catalog))
    if args.limit:
        cards = cards[: args.limit]
    if not cards:
        raise SystemExit(f"No cards found in {args.catalog}. Run scripts/import_official_cards.py first.")

    preprocessor = CardPreprocessor()
    embedder = ImageEmbedder(model_name=args.model)
    rows = []
    start = time.perf_counter()
    for index, card in enumerate(cards, start=1):
        if not card.local_image_path:
            print(f"skip {card.card_id}: no local image")
            continue
        image_path = ROOT / card.local_image_path
        if not image_path.exists():
            print(f"skip {card.card_id}: missing {card.local_image_path}")
            continue
        item_start = time.perf_counter()
        processed = preprocessor.preprocess(image_path, use_detector=False)
        vector = embedder.embed_image(processed.image)
        rows.append(
            CardVectorStore.card_to_row(
                card,
                vector,
                embedding_model=args.model,
                preprocess_version=PREPROCESS_VERSION,
            )
        )
        elapsed = (time.perf_counter() - item_start) * 1000
        print(f"{index}/{len(cards)} {card.card_id} {card.name} {elapsed:.1f} ms")

    if not rows:
        raise SystemExit("No vectors were generated.")
    count = CardVectorStore(args.db, args.table).create(rows, mode="overwrite")
    total = time.perf_counter() - start
    print(f"Indexed {count} cards in {total:.2f}s at {args.db}/{args.table}")


if __name__ == "__main__":
    main()
