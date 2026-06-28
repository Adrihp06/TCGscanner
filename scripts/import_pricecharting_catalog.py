from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pricecharting_cards.json"
IMAGE_ROOT = ROOT / "images" / "pricecharting"
BASE_URL = "https://www.pricecharting.com"

SETS = {
    "OGN": {
        "slug": "riftbound-origins",
        "name": "Origins",
        "url": "https://www.pricecharting.com/console/riftbound-origins?sort=model-number",
    },
    "SFG": {
        "slug": "riftbound-spiritforged",
        "name": "Spiritforged",
        "url": "https://www.pricecharting.com/console/riftbound-spiritforged?sort=model-number",
    },
    "UNL": {
        "slug": "riftbound-unleashed",
        "name": "Unleashed",
        "url": "https://www.pricecharting.com/console/riftbound-unleashed?sort=model-number",
    },
}


def fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RiftboundScannerMVP/0.1 (+local dataset import)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "RiftboundScannerMVP/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())


def parse_money(value: str) -> float | None:
    cleaned = re.sub(r"[^0-9.]", "", html.unescape(value))
    return float(cleaned) if cleaned else None


def clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_title(title: str) -> tuple[str, str]:
    match = re.search(r"\s+#([0-9]+[a-zA-Z]?)\s*$", title)
    if not match:
        return title, ""
    number = match.group(1)
    name = title[: match.start()].strip()
    return name, number


def parse_rows(page: str, set_code: str, set_meta: dict[str, str]) -> list[dict[str, object]]:
    rows = re.findall(r"<tr[^>]*data-product=\"([^\"]+)\"[^>]*>(.*?)</tr>", page, re.S)
    cards: list[dict[str, object]] = []
    for product_id, row in rows:
        link = re.search(r'<td class="title"[^>]*>\s*<a href="([^"]+)">(.*?)</a>', row, re.S)
        image = re.search(r'<img class="photo"[^>]*src="([^"]+)"', row)
        price = re.search(
            r'<td class="price numeric used_price"[^>]*>.*?<span[^>]*class="[^"]*js-price[^"]*"[^>]*>\s*([^<]*)\s*</span>',
            row,
            re.S,
        )
        if not link:
            continue

        href = html.unescape(link.group(1))
        if href.startswith("/"):
            href = f"{BASE_URL}{href}"
        title = clean(link.group(2))
        name, collector = parse_title(title)
        if not collector:
            continue

        image_url = html.unescape(image.group(1)) if image else None
        high_res = image_url.replace("/60.jpg", "/240.jpg") if image_url else None
        cards.append(
            {
                "card_id": f"{set_code}-{collector.upper()}-{product_id}",
                "name": name,
                "set_code": set_code,
                "set_name": set_meta["name"],
                "printed_number": f"{collector.lower()}/298" if set_code == "OGN" else collector.lower(),
                "collector_number": collector.lower(),
                "language": "English",
                "language_code": "EN",
                "cardmarket_product_id": None,
                "pricecharting_url": href,
                "pricecharting_set_slug": set_meta["slug"],
                "pricecharting_product_id": product_id,
                "pricecharting_ungraded_usd": parse_money(price.group(1)) if price else None,
                "image_url": high_res,
                "source_title": title,
            }
        )
    return cards


def import_catalog(limit_images: int) -> dict[str, object]:
    all_cards: list[dict[str, object]] = []
    for set_code, set_meta in SETS.items():
        print(f"Fetching {set_meta['name']}...")
        page = fetch(set_meta["url"])
        cards = parse_rows(page, set_code, set_meta)
        all_cards.extend(cards)

        image_dir = IMAGE_ROOT / set_meta["slug"]
        image_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for card in cards:
            if downloaded >= limit_images:
                break
            if "[Foil]" in str(card["name"]):
                continue
            image_url = card.get("image_url")
            if not image_url:
                continue
            safe_number = re.sub(r"[^0-9a-zA-Z]+", "_", str(card["collector_number"]))
            safe_name = re.sub(r"[^0-9a-zA-Z]+", "_", str(card["name"])).strip("_").lower()
            path = image_dir / f"{safe_number}_{safe_name}.jpg"
            print(f"  image {path.relative_to(ROOT)}")
            download(str(image_url), path)
            card["local_image_path"] = str(path.relative_to(ROOT))
            downloaded += 1
            time.sleep(0.1)

    payload = {
        "source": "pricecharting",
        "sets": SETS,
        "cards": all_cards,
    }
    DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Riftbound catalog/images from PriceCharting.")
    parser.add_argument("--limit-images", type=int, default=10)
    args = parser.parse_args()
    payload = import_catalog(args.limit_images)
    print(f"Imported {len(payload['cards'])} cards into {DATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
