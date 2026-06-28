from __future__ import annotations

import argparse
import json
from pathlib import Path

from riftbound_scanner.pricing import default_price_provider
from riftbound_scanner.visual import DEFAULT_MODEL, VisualCardMatcher

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Riftbound visual card index with one image.")
    parser.add_argument("image", help="Path to a card photo.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--language", default="EN")
    parser.add_argument("--seller-country", default="ES")
    parser.add_argument("--price-mode", choices=["min", "trend"], default="min")
    parser.add_argument("--db", default=str(ROOT / "data" / "vector_db"))
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    matcher = VisualCardMatcher(args.db, args.table, args.model)
    result = matcher.search(args.image, top_k=args.top_k)
    best = result.matches[0] if result.matches else None
    price = (
        default_price_provider().get_price(
            best.card,
            args.language.upper(),
            args.seller_country.upper(),
            args.price_mode,
        )
        if best
        else None
    )

    payload = {
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
        "best_match": None
        if best is None
        else {
            "score": best.score,
            "distance": best.distance,
            "card_id": best.card.card_id,
            "name": best.card.name,
            "set_code": best.card.set_code,
            "printed_number": best.card.printed_number,
        },
        "price": None
        if price is None
        else {
            "mode": price.mode,
            "amount": price.amount,
            "currency": price.currency,
            "source": price.source,
            "message": price.message,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
