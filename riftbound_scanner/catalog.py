from __future__ import annotations

import csv
import difflib
import json
import re
from dataclasses import replace
from pathlib import Path

from .models import Card, CardIdentity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CATALOG_PATH = DATA_DIR / "riftbound_cards.json"
PRICECHARTING_CATALOG_PATH = DATA_DIR / "pricecharting_cards.json"
OFFICIAL_CATALOG_PATH = DATA_DIR / "official_cards.json"
OFFICIAL_UNLEASHED_PATH = DATA_DIR / "unleashed_cards.json"


class CardCatalog:
    def __init__(self, cards: list[Card]) -> None:
        self.cards = cards
        self._by_identity: dict[str, Card] = {}
        for card in cards:
            self._by_identity.setdefault(card.identity.key, card)

    @classmethod
    def from_json(cls, path: Path = DEFAULT_CATALOG_PATH) -> "CardCatalog":
        cards: list[Card] = []
        for source in [path, OFFICIAL_CATALOG_PATH, OFFICIAL_UNLEASHED_PATH, PRICECHARTING_CATALOG_PATH]:
            if not source.exists():
                continue
            payload = json.loads(source.read_text(encoding="utf-8"))
            cards.extend(cls._card_from_dict(item) for item in payload["cards"])
        by_key: dict[str, Card] = {}
        for card in cards:
            key = f"{card.set_code}:{card.normalized_collector_number}:{card.name}"
            existing = by_key.get(key)
            by_key[key] = cls._merge_card(existing, card) if existing else card
        return cls(list(by_key.values()))

    def resolve(self, identity: CardIdentity) -> Card | None:
        normalized = self._normalize_identity(identity)
        exact = self._by_identity.get(normalized.key)
        if exact:
            return exact
        collector = self._collector_from_printed(normalized.printed_number)
        for card in self.cards:
            if card.set_code.upper() == normalized.set_code.upper() and card.normalized_collector_number == collector:
                return card
        return None

    def resolve_fuzzy(
        self,
        identity: CardIdentity,
        card_name: str | None = None,
        raw_text: str | None = None,
    ) -> tuple[Card | None, float, str | None]:
        exact = self.resolve(identity)
        if exact:
            return exact, 1.0, None

        normalized = self._normalize_identity(identity)
        collector = self._collector_from_printed(normalized.printed_number)
        set_code = normalized.set_code.upper()
        name_hint = (card_name or raw_text or "").casefold()
        candidate_set_codes = self._candidate_set_codes(set_code, collector)
        candidates = [card for card in self.cards if card.set_code.upper() in candidate_set_codes]
        scored: list[tuple[float, Card, str]] = []
        for card in candidates:
            number_score = self._collector_similarity(collector, card.normalized_collector_number)
            if self._looks_like_suffix_confusion(collector) and card.normalized_collector_number.endswith("b"):
                number_score = max(number_score, 0.98)
            name_score = difflib.SequenceMatcher(None, name_hint, card.name.casefold()).ratio() if name_hint else 0
            score = number_score * 0.72 + name_score * 0.28
            if number_score >= 0.72 or (number_score >= 0.55 and name_score >= 0.45):
                reason = f"matched collector {collector!r} to catalog {card.normalized_collector_number!r}"
                scored.append((score, card, reason))
        if not scored:
            return None, 0.0, None
        scored.sort(key=lambda item: item[0], reverse=True)
        score, card, reason = scored[0]
        return card, score, reason

    def search(self, query: str) -> list[Card]:
        needle = query.casefold().strip()
        if not needle:
            return self.cards
        return [
            card
            for card in self.cards
            if needle in card.name.casefold()
            or needle in card.set_code.casefold()
            or needle in card.printed_number.casefold()
            or needle in card.normalized_collector_number.casefold()
        ]

    def export_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "card_id",
                    "name",
                    "set_code",
                    "set_name",
                    "printed_number",
                    "language_code",
                    "cardmarket_product_id",
                    "pricecharting_url",
                ],
            )
            writer.writeheader()
            for card in self.cards:
                writer.writerow(
                    {
                        "card_id": card.card_id,
                        "name": card.name,
                        "set_code": card.set_code,
                        "set_name": card.set_name,
                        "printed_number": card.printed_number,
                        "language_code": card.language_code,
                        "cardmarket_product_id": card.cardmarket_product_id or "",
                        "pricecharting_url": card.pricecharting_url or "",
                    }
                )

    @staticmethod
    def _card_from_dict(item: dict[str, object]) -> Card:
        allowed = set(Card.__dataclass_fields__.keys())
        payload = {key: value for key, value in item.items() if key in allowed}
        if payload.get("collector_number") is None:
            payload["collector_number"] = str(payload["printed_number"]).split("/", 1)[0]
        return Card(**payload)

    @staticmethod
    def _merge_card(existing: Card, incoming: Card) -> Card:
        return replace(
            existing,
            cardmarket_product_id=existing.cardmarket_product_id or incoming.cardmarket_product_id,
            pricecharting_url=existing.pricecharting_url or incoming.pricecharting_url,
            pricecharting_set_slug=existing.pricecharting_set_slug or incoming.pricecharting_set_slug,
            pricecharting_product_id=existing.pricecharting_product_id or incoming.pricecharting_product_id,
            pricecharting_ungraded_usd=(
                existing.pricecharting_ungraded_usd
                if existing.pricecharting_ungraded_usd is not None
                else incoming.pricecharting_ungraded_usd
            ),
        )

    @staticmethod
    def _normalize_identity(identity: CardIdentity) -> CardIdentity:
        printed = identity.printed_number.lower().strip()
        match = re.search(r"(\d+[a-z]?)(?:\s*/\s*(\d+))?", printed, re.I)
        if match:
            number = match.group(1).lower().lstrip("0") or match.group(1).lower()
            printed = f"{number}/{match.group(2)}" if match.group(2) else number
        return CardIdentity(identity.set_code.upper().strip(), printed)

    @staticmethod
    def _collector_from_printed(printed_number: str) -> str:
        return printed_number.split("/", 1)[0].lower().lstrip("0") or printed_number.lower()

    @staticmethod
    def _collector_similarity(query_value: str, catalog_value: str) -> float:
        normalized_query = CardCatalog._collector_confusion_normalize(query_value)
        normalized_catalog = CardCatalog._collector_confusion_normalize(catalog_value)
        if normalized_query == normalized_catalog:
            return 1.0
        if CardCatalog._has_variant_suffix(query_value) and not CardCatalog._has_variant_suffix(catalog_value):
            return 0.25
        ratio = difflib.SequenceMatcher(None, normalized_query, normalized_catalog).ratio()
        if normalized_query[:-1] == normalized_catalog[:-1] and {normalized_query[-1:], normalized_catalog[-1:]} <= {"0", "b", "s", "5"}:
            ratio = max(ratio, 0.96)
        return ratio

    @staticmethod
    def _collector_confusion_normalize(value: str) -> str:
        value = value.lower().strip()
        if value.endswith("0"):
            return f"{value[:-1]}b"
        if value.endswith("s") or value.endswith("5"):
            return f"{value[:-1]}b"
        return value

    @staticmethod
    def _looks_like_suffix_confusion(value: str) -> bool:
        return value.lower().endswith(("0", "s", "5"))

    @staticmethod
    def _candidate_set_codes(set_code: str, collector: str) -> set[str]:
        codes = {set_code.upper()}
        if set_code.upper() == "OGN" and (
            CardCatalog._looks_like_suffix_confusion(collector)
            or CardCatalog._has_variant_suffix(collector)
        ):
            codes.update({"PRO", "PR", "PROMO"})
        return codes

    @staticmethod
    def _has_variant_suffix(value: str) -> bool:
        return bool(value) and value[-1:].isalpha()
