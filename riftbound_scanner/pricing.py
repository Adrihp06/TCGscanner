from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .models import Card, Price, PriceMode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FIXTURE_PATH = DATA_DIR / "prices_fixture.json"


class PriceProvider(Protocol):
    def get_price(
        self,
        card: Card,
        language: str,
        seller_country: str,
        mode: PriceMode,
    ) -> Price:
        ...


class FixturePriceProvider:
    def __init__(self, path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self.path = path
        self.payload = json.loads(path.read_text(encoding="utf-8"))

    def get_price(
        self,
        card: Card,
        language: str,
        seller_country: str,
        mode: PriceMode,
    ) -> Price:
        row = self.payload.get(card.card_id)
        filters = {"language": language, "seller_country": seller_country}
        if not row:
            return Price(
                mode=mode,
                amount=None,
                source="fixture",
                filters=filters,
                message="No fixture price for this card.",
            )
        return Price(
            mode=mode,
            amount=float(row.get(mode, 0)),
            currency=row.get("currency", "EUR"),
            source="fixture",
            filters=filters,
        )


def default_price_provider() -> PriceProvider:
    provider = os.getenv("RIFTBOUND_PRICE_PROVIDER", "pricecharting").lower()
    if provider == "cardmarket":
        from .cardmarket import CardmarketPriceProvider

        return CardmarketPriceProvider.from_env()
    if provider == "pricecharting":
        from .pricecharting import PriceChartingPriceProvider

        return PriceChartingPriceProvider()
    return FixturePriceProvider()
