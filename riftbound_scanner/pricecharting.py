from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import Card, Price, PriceMode

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "pricecharting"
BASE_URL = "https://www.pricecharting.com"

SET_SLUGS = {
    "OGN": "riftbound-origins",
    "UNL": "riftbound-unleashed",
    "SFG": "riftbound-spiritforged",
    "PRO": "riftbound-promo",
    "PR": "riftbound-promo",
    "PROMO": "riftbound-promo",
}


@dataclass(frozen=True)
class PriceChartingMatch:
    title: str
    url: str
    amount: float | None


class PriceChartingPriceProvider:
    def __init__(self, ttl_seconds: int = 60 * 60 * 24) -> None:
        self.ttl_seconds = ttl_seconds
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_price(
        self,
        card: Card,
        language: str,
        seller_country: str,
        mode: PriceMode,
    ) -> Price:
        match = self._find_match(card)
        filters = {
            "language": language,
            "seller_country": seller_country,
            "pricing_basis": "PriceCharting ungraded value",
        }
        if not match:
            return Price(
                mode=mode,
                amount=None,
                currency="USD",
                source="pricecharting",
                filters=filters,
                message="No PriceCharting match found.",
            )
        return Price(
            mode=mode,
            amount=match.amount,
            currency="USD",
            source="pricecharting",
            filters={**filters, "url": match.url, "matched_title": match.title},
            message=(
                "PriceCharting does not expose Cardmarket-style min/trend; "
                "this is the current ungraded value."
            ),
        )

    def _find_match(self, card: Card) -> PriceChartingMatch | None:
        if card.pricecharting_ungraded_usd is not None and card.pricecharting_url:
            return PriceChartingMatch(
                title=card.source_title or card.name,
                url=card.pricecharting_url,
                amount=card.pricecharting_ungraded_usd,
            )
        if card.pricecharting_url:
            amount = self._fetch_product_price(card.pricecharting_url)
            return PriceChartingMatch(card.name, card.pricecharting_url, amount)

        query = self._query_for_card(card)
        url = f"{BASE_URL}/search-products?{urllib.parse.urlencode({'q': query, 'type': 'prices'})}"
        page = self._fetch_url(url)
        canonical = self._canonical_product_url(page)
        if canonical:
            amount = self._parse_product_price(page)
            title = self._parse_product_title(page) or card.name
            return PriceChartingMatch(title=title, url=canonical, amount=amount)

        candidates = self._parse_search_candidates(page)
        selected = self._select_candidate(card, candidates)
        if not selected:
            return None
        if selected.amount is not None:
            return selected
        amount = self._fetch_product_price(selected.url)
        return PriceChartingMatch(selected.title, selected.url, amount)

    def _query_for_card(self, card: Card) -> str:
        printed = card.printed_number.split("/", 1)[0].lstrip("0") or card.printed_number
        return f"{card.name} {printed}".strip()

    def _select_candidate(
        self, card: Card, candidates: list[PriceChartingMatch]
    ) -> PriceChartingMatch | None:
        number = card.printed_number.split("/", 1)[0].lstrip("0").casefold()
        name_tokens = set(re.findall(r"[a-z0-9]+", card.name.casefold()))
        wanted_set = card.pricecharting_set_slug or SET_SLUGS.get(card.set_code.upper())

        scored: list[tuple[int, PriceChartingMatch]] = []
        for candidate in candidates:
            title = candidate.title.casefold()
            url = candidate.url.casefold()
            score = 0
            if wanted_set and f"/game/{wanted_set}/" in url:
                score += 8
            if f"#{number}" in title or title.endswith(f" {number}"):
                score += 6
            score += sum(1 for token in name_tokens if token in title)
            if score:
                scored.append((score, candidate))
        if not scored:
            return candidates[0] if len(candidates) == 1 else None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _fetch_product_price(self, url: str) -> float | None:
        page = self._fetch_url(url)
        return self._parse_product_price(page)

    def _fetch_url(self, url: str) -> str:
        cache_path = CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()}.html"
        if cache_path.exists() and time.time() - cache_path.stat().st_mtime < self.ttl_seconds:
            return cache_path.read_text(encoding="utf-8")

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RiftboundScannerMVP/0.1 (+local development)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
        cache_path.write_text(body, encoding="utf-8")
        return body

    def _canonical_product_url(self, page: str) -> str | None:
        match = re.search(r'<link rel="canonical" href="([^"]+/game/riftbound-[^"]+)">', page)
        return html.unescape(match.group(1)) if match else None

    def _parse_product_title(self, page: str) -> str | None:
        match = re.search(r'<h1[^>]*id="product_name"[^>]*>(.*?)</h1>', page, re.S)
        if not match:
            return None
        return self._clean_text(match.group(1).split("<a", 1)[0])

    def _parse_product_price(self, page: str) -> float | None:
        match = re.search(
            r'<td[^>]*id="used_price"[^>]*>.*?<span[^>]*class="[^"]*js-price[^"]*"[^>]*>\s*([^<]+)\s*</span>',
            page,
            re.S,
        )
        if not match:
            return None
        return self._parse_money(match.group(1))

    def _parse_search_candidates(self, page: str) -> list[PriceChartingMatch]:
        rows = re.findall(r"<tr[^>]*data-product=\"[^\"]+\"[^>]*>(.*?)</tr>", page, re.S)
        candidates: list[PriceChartingMatch] = []
        for row in rows:
            link = re.search(r'<td class="title"[^>]*>\s*<a href="([^"]+)">(.*?)</a>', row, re.S)
            if not link:
                continue
            price = re.search(
                r'<td class="price numeric used_price"[^>]*>.*?<span[^>]*class="[^"]*js-price[^"]*"[^>]*>\s*([^<]*)\s*</span>',
                row,
                re.S,
            )
            href = html.unescape(link.group(1))
            if href.startswith("/"):
                href = f"{BASE_URL}{href}"
            candidates.append(
                PriceChartingMatch(
                    title=self._clean_text(link.group(2)),
                    url=href,
                    amount=self._parse_money(price.group(1)) if price else None,
                )
            )
        return candidates

    @staticmethod
    def _clean_text(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = html.unescape(value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _parse_money(value: str) -> float | None:
        cleaned = re.sub(r"[^0-9.]", "", html.unescape(value))
        if not cleaned:
            return None
        return float(cleaned)
