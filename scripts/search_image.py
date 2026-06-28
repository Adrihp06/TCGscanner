from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.visual import DEFAULT_MODEL, VisualCardMatcher

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the visual card index with one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--db", default=str(ROOT / "data" / "vector_db"))
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    matcher = VisualCardMatcher(args.db, args.table, args.model)
    results = [matcher.search(args.image, top_k=args.top_k) for _ in range(max(1, args.repeat))]
    result = results[-1]
    payload = {
        "runs": len(results),
        "all_latency_ms": [item.latency_ms for item in results],
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
        "matches": [
            {
                "rank": match.rank,
                "score": match.score,
                "distance": match.distance,
                "card_id": match.card.card_id,
                "name": match.card.name,
                "printed_number": match.card.printed_number,
            }
            for match in result.matches
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
