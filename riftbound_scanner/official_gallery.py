from __future__ import annotations

import html
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from .models import Card

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_ROOT = ROOT / "images" / "official"
OFFICIAL_GALLERY_URL = "https://riftbound.leagueoflegends.com/en-us/card-gallery/"
DEFAULT_OFFICIAL_PATH = DATA_DIR / "official_cards.json"
DEFAULT_UNLEASHED_PATH = DATA_DIR / "unleashed_cards.json"
PRICECHARTING_SET_SLUGS = {
    "OGN": "riftbound-origins",
    "SFD": "riftbound-spiritforged",
    "UNL": "riftbound-unleashed",
}


def fetch_official_gallery(url: str = OFFICIAL_GALLERY_URL) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "RiftboundScanner/visual-index"})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read().decode("utf-8", errors="replace")
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', body)
    if not match:
        raise ValueError("Official gallery payload was not found.")
    return json.loads(html.unescape(match.group(1)))


def extract_card_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for blade in payload.get("props", {}).get("pageProps", {}).get("page", {}).get("blades", []):
        cards = blade.get("cards")
        if isinstance(cards, dict) and isinstance(cards.get("items"), list):
            return cards["items"]
    raise ValueError("Official gallery card list was not found.")


def available_sets(payload: dict[str, Any]) -> list[dict[str, object]]:
    sets: dict[str, dict[str, object]] = {}
    for item in extract_card_items(payload):
        set_value = item.get("set", {}).get("value", {})
        set_code = str(set_value.get("id") or "").upper()
        if not set_code:
            continue
        entry = sets.setdefault(
            set_code,
            {
                "set_code": set_code,
                "set_name": str(set_value.get("label") or set_code),
                "card_count": 0,
                "image_directory": f"images/official/{slugify(str(set_value.get('label') or set_code))}",
            },
        )
        entry["card_count"] = int(entry["card_count"]) + 1
    return [sets[set_code] for set_code in sorted(sets)]


def extract_cards(payload: dict[str, Any], set_codes: str | list[str] | None = "UNL") -> list[dict[str, Any]]:
    items = extract_card_items(payload)
    if set_codes is None:
        return items
    if isinstance(set_codes, str):
        wanted = {set_codes.upper()}
    else:
        wanted = {set_code.upper() for set_code in set_codes}
    return [
        item
        for item in items
        if str(item.get("set", {}).get("value", {}).get("id") or "").upper() in wanted
    ]


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-").lower()
    return slug or "unknown-set"


def filename_component(value: str) -> str:
    value = value.replace("*", "_star")
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower() or "unknown"


def collector_from_public_code(public_code: str, fallback: object) -> str:
    if "-" in public_code:
        printed = public_code.split("-", 1)[1]
        return printed.split("/", 1)[0].strip().lower()
    return str(fallback or "").strip().lower()


def official_card_to_dict(item: dict[str, Any]) -> dict[str, object]:
    set_value = item.get("set", {}).get("value", {})
    set_code = str(set_value.get("id") or "").upper()
    public_code = str(item.get("publicCode") or "").strip()
    collector = collector_from_public_code(public_code, item.get("collectorNumber"))
    printed = public_code.split("-", 1)[1].lower() if "-" in public_code else collector.lower()
    image_url = str(item.get("cardImage", {}).get("url") or "")
    return {
        "card_id": str(item.get("id") or f"{set_code}-{collector}").upper(),
        "name": str(item.get("name") or "").strip(),
        "set_code": set_code,
        "set_name": str(set_value.get("label") or set_code),
        "printed_number": printed,
        "collector_number": collector.lower(),
        "language": "English",
        "language_code": "EN",
        "cardmarket_product_id": None,
        "pricecharting_url": None,
        "pricecharting_set_slug": PRICECHARTING_SET_SLUGS.get(set_code),
        "pricecharting_product_id": None,
        "pricecharting_ungraded_usd": None,
        "image_url": image_url,
        "source_title": str(item.get("publicCode") or item.get("name") or ""),
    }


def download_image(url: str, path: Path, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "RiftboundScanner/visual-index"})
    with urllib.request.urlopen(request, timeout=45) as response:
        path.write_bytes(response.read())
    return True


def local_image_path(card: dict[str, object]) -> Path:
    set_slug = slugify(str(card["set_name"]))
    collector = filename_component(str(card["collector_number"]))
    name = filename_component(str(card["name"]))
    return IMAGE_ROOT / set_slug / f"{collector}_{name}.png"


def import_official_cards(
    output_path: Path = DEFAULT_OFFICIAL_PATH,
    set_codes: list[str] | None = None,
    limit: int = 0,
    download_images: bool = True,
    overwrite_images: bool = False,
    sleep_seconds: float = 0.03,
) -> dict[str, object]:
    payload = fetch_official_gallery()
    raw_cards = extract_cards(payload, set_codes)
    if limit > 0:
        raw_cards = raw_cards[:limit]

    cards: list[dict[str, object]] = []
    for raw in raw_cards:
        card = official_card_to_dict(raw)
        image_url = str(card.get("image_url") or "")
        if download_images and image_url:
            path = local_image_path(card)
            downloaded = download_image(image_url, path, overwrite=overwrite_images)
            card["local_image_path"] = path.relative_to(ROOT).as_posix()
            if downloaded and sleep_seconds:
                time.sleep(sleep_seconds)
        cards.append(card)

    set_summaries: dict[str, dict[str, object]] = {}
    for card in cards:
        set_code = str(card["set_code"])
        entry = set_summaries.setdefault(
            set_code,
            {
                "set_code": set_code,
                "set_name": str(card["set_name"]),
                "card_count": 0,
                "image_directory": f"images/official/{slugify(str(card['set_name']))}",
            },
        )
        entry["card_count"] = int(entry["card_count"]) + 1
    output = {
        "source": "official_riftbound_gallery",
        "source_url": OFFICIAL_GALLERY_URL,
        "set_codes": sorted(set_summaries),
        "sets": [set_summaries[set_code] for set_code in sorted(set_summaries)],
        "available_sets": available_sets(payload),
        "cards": cards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def import_unleashed(
    output_path: Path = DEFAULT_UNLEASHED_PATH,
    limit: int = 0,
    download_images: bool = True,
    overwrite_images: bool = False,
    sleep_seconds: float = 0.03,
) -> dict[str, object]:
    return import_official_cards(
        output_path=output_path,
        set_codes=["UNL"],
        limit=limit,
        download_images=download_images,
        overwrite_images=overwrite_images,
        sleep_seconds=sleep_seconds,
    )


def load_official_cards(path: Path = DEFAULT_OFFICIAL_PATH) -> list[Card]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(Card.__dataclass_fields__.keys())
    return [Card(**{key: value for key, value in item.items() if key in allowed}) for item in payload["cards"]]


def load_unleashed_cards(path: Path = DEFAULT_UNLEASHED_PATH) -> list[Card]:
    return load_official_cards(path)
