from __future__ import annotations

import argparse

from riftbound_scanner.catalog import CardCatalog
from riftbound_scanner.pricing import default_price_provider
from riftbound_scanner.vision import VisionScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a Riftbound card image.")
    parser.add_argument("image", help="Path to a card photo.")
    parser.add_argument("--language", default="EN", help="Card language code.")
    parser.add_argument("--seller-country", default="ES", help="Cardmarket seller country code.")
    parser.add_argument("--price-mode", choices=["min", "trend"], default="min")
    args = parser.parse_args()

    extraction = VisionScanner().extract_from_image(args.image)
    print(f"OCR text: {extraction.raw_text or '<empty>'}")
    for warning in extraction.warnings:
        print(f"Warning: {warning}")
    if extraction.identity is None:
        raise SystemExit(2)

    catalog = CardCatalog.from_json()
    card = catalog.resolve(extraction.identity)
    if card is None:
        print(f"Card not found: {extraction.identity.set_code} {extraction.identity.printed_number}")
        raise SystemExit(1)

    price = default_price_provider().get_price(
        card,
        args.language.upper(),
        args.seller_country.upper(),
        args.price_mode,
    )
    amount = "unavailable" if price.amount is None else f"{price.amount:.2f} {price.currency}"
    print(f"{card.name} ({card.set_code} {card.printed_number})")
    print(f"{price.mode}: {amount} from {price.source}")


if __name__ == "__main__":
    main()
