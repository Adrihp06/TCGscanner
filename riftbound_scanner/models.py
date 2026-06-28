from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

PriceMode = Literal["min", "trend"]


@dataclass(frozen=True)
class CardIdentity:
    set_code: str
    printed_number: str

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.printed_number}"


@dataclass(frozen=True)
class Card:
    card_id: str
    name: str
    set_code: str
    set_name: str
    printed_number: str
    collector_number: str | None = None
    language: str = "English"
    language_code: str = "EN"
    cardmarket_product_id: int | None = None
    pricecharting_url: str | None = None
    pricecharting_set_slug: str | None = None
    pricecharting_product_id: str | None = None
    pricecharting_ungraded_usd: float | None = None
    local_image_path: str | None = None
    source_title: str | None = None
    image_url: str | None = None

    @property
    def identity(self) -> CardIdentity:
        return CardIdentity(self.set_code, self.printed_number)

    @property
    def normalized_collector_number(self) -> str:
        value = self.collector_number or self.printed_number.split("/", 1)[0]
        return value.lower().lstrip("0") or value.lower()


@dataclass(frozen=True)
class Price:
    mode: PriceMode
    amount: float | None
    currency: str = "EUR"
    source: str = "fixture"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    filters: dict[str, str] = field(default_factory=dict)
    message: str | None = None


@dataclass(frozen=True)
class ResolveResult:
    card: Card | None
    confidence: float
    price: Price | None
    input: dict[str, str]
    warnings: list[str] = field(default_factory=list)
