from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.official_gallery import DEFAULT_UNLEASHED_PATH, load_unleashed_cards
from riftbound_scanner.visual import DEFAULT_MODEL, VisualCardMatcher

ROOT = Path(__file__).resolve().parent.parent


def augment(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = image.convert("RGB")
    angle = rng.uniform(-8, 8)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(238, 240, 242))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.2))
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))
    border = rng.randint(18, 72)
    return ImageOps.expand(image, border=border, fill=(rng.randint(20, 240), rng.randint(20, 240), rng.randint(20, 240)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visual retrieval precision and latency.")
    parser.add_argument("--catalog", default=str(DEFAULT_UNLEASHED_PATH))
    parser.add_argument("--db", default=str(ROOT / "data" / "vector_db"))
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default=str(ROOT / "reports" / "visual_retrieval_eval.json"))
    args = parser.parse_args()

    cards = [card for card in load_unleashed_cards(Path(args.catalog)) if card.local_image_path]
    rng = random.Random(args.seed)
    rng.shuffle(cards)
    cards = cards[: args.samples]
    if not cards:
        raise SystemExit("No local card images found for evaluation.")

    matcher = VisualCardMatcher(args.db, args.table, args.model)
    rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, card in enumerate(cards, start=1):
            source = ROOT / str(card.local_image_path)
            query_path = Path(temp_dir) / f"{card.card_id}.jpg"
            augment(Image.open(source), args.seed + index).save(query_path, quality=82)
            start = time.perf_counter()
            result = matcher.search(query_path, top_k=args.top_k)
            wall_ms = (time.perf_counter() - start) * 1000
            match_ids = [match.card.card_id for match in result.matches]
            rank = match_ids.index(card.card_id) + 1 if card.card_id in match_ids else None
            rows.append(
                {
                    "card_id": card.card_id,
                    "name": card.name,
                    "expected_rank": rank,
                    "top1": match_ids[0] if match_ids else None,
                    "latency_ms": result.latency_ms,
                    "wall_ms": round(wall_ms, 2),
                    "matches": [
                        {
                            "rank": match.rank,
                            "card_id": match.card.card_id,
                            "name": match.card.name,
                            "score": match.score,
                            "distance": match.distance,
                        }
                        for match in result.matches
                    ],
                }
            )
            print(f"{index}/{len(cards)} {card.card_id} rank={rank} wall={wall_ms:.1f}ms")

    precision_at_1 = sum(1 for row in rows if row["expected_rank"] == 1) / len(rows)
    precision_at_3 = sum(1 for row in rows if row["expected_rank"] and row["expected_rank"] <= 3) / len(rows)
    mrr = sum(1 / row["expected_rank"] for row in rows if row["expected_rank"]) / len(rows)
    latencies = sorted(row["wall_ms"] for row in rows)
    payload = {
        "samples": len(rows),
        "precision_at_1": precision_at_1,
        "precision_at_3": precision_at_3,
        "mrr": mrr,
        "latency_wall_ms": {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
