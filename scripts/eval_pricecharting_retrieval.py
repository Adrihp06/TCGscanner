from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riftbound_scanner.visual import DEFAULT_MODEL, VisualCardMatcher

ROOT = Path(__file__).resolve().parent.parent
SET_SLUG_TO_CODE = {
    "riftbound-origins": "OGN",
    "riftbound-spiritforged": "SFD",
    "riftbound-unleashed": "UNL",
}


def expected_from_path(path: Path) -> tuple[str, str]:
    set_code = SET_SLUG_TO_CODE.get(path.parent.name)
    if not set_code:
        raise ValueError(f"Unsupported PriceCharting set directory: {path.parent.name}")
    match = re.match(r"([0-9]+[a-zA-Z]?)(?:_.*)?$", path.stem)
    if not match:
        raise ValueError(f"Could not parse collector number from {path.name}")
    collector = match.group(1).lower().lstrip("0") or match.group(1).lower()
    return set_code, collector


def normalized_collector(value: str) -> str:
    return value.lower().split("/", 1)[0].lstrip("0") or value.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate visual retrieval using local PriceCharting images as external queries."
    )
    parser.add_argument("--root", default=str(ROOT / "images" / "pricecharting"))
    parser.add_argument("--db", default=str(ROOT / "data" / "vector_db"))
    parser.add_argument("--table", default="card_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "reports" / "pricecharting_retrieval_eval.json"))
    args = parser.parse_args()

    paths = sorted(
        path
        for path in Path(args.root).glob("*/*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No PriceCharting images found under {args.root}")

    matcher = VisualCardMatcher(args.db, args.table, args.model)
    rows = []
    by_set: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, path in enumerate(paths, start=1):
        expected_set, expected_collector = expected_from_path(path)
        result = matcher.search(path, top_k=args.top_k)
        rank = None
        for match in result.matches:
            if (
                match.card.set_code == expected_set
                and normalized_collector(match.card.collector_number or match.card.printed_number) == expected_collector
            ):
                rank = match.rank
                break
        top = result.matches[0] if result.matches else None
        row = {
            "query": path.relative_to(ROOT).as_posix(),
            "expected_set": expected_set,
            "expected_collector": expected_collector,
            "expected_rank": rank,
            "top1_card_id": top.card.card_id if top else None,
            "top1_name": top.card.name if top else None,
            "top1_score": top.score if top else None,
            "latency_ms": result.latency_ms,
            "matches": [
                {
                    "rank": match.rank,
                    "card_id": match.card.card_id,
                    "set_code": match.card.set_code,
                    "collector_number": normalized_collector(match.card.collector_number or match.card.printed_number),
                    "name": match.card.name,
                    "score": match.score,
                    "distance": match.distance,
                }
                for match in result.matches
            ],
        }
        rows.append(row)
        by_set[expected_set].append(row)
        print(
            f"{index}/{len(paths)} {path.parent.name}/{path.name} expected={expected_set}-{expected_collector} rank={rank}"
        )

    def metrics(items: list[dict[str, object]]) -> dict[str, float | int]:
        count = len(items)
        return {
            "samples": count,
            "precision_at_1": sum(1 for item in items if item["expected_rank"] == 1) / count if count else 0.0,
            "precision_at_3": sum(
                1 for item in items if item["expected_rank"] and int(item["expected_rank"]) <= 3
            )
            / count
            if count
            else 0.0,
            "mrr": sum(1 / int(item["expected_rank"]) for item in items if item["expected_rank"]) / count
            if count
            else 0.0,
        }

    latencies = sorted(float(row["latency_ms"]["total"]) for row in rows)
    payload = {
        **metrics(rows),
        "latency_total_ms": {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        },
        "by_set": {set_code: metrics(items) for set_code, items in sorted(by_set.items())},
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
