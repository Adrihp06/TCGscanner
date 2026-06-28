import unittest

from riftbound_scanner.catalog import CardCatalog
from riftbound_scanner.models import CardIdentity
from riftbound_scanner.pricing import FixturePriceProvider


class CatalogPricingTest(unittest.TestCase):
    def test_resolve_card(self):
        catalog = CardCatalog.from_json()
        card = catalog.resolve(CardIdentity("OGN", "045/298"))
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "Defy")

    def test_resolve_alter(self):
        catalog = CardCatalog.from_json()
        card = catalog.resolve(CardIdentity("UNL", "001a/260"))
        self.assertIsNotNone(card)
        self.assertEqual(card.card_id, "UNL-001A")

    def test_fixture_price(self):
        catalog = CardCatalog.from_json()
        card = catalog.resolve(CardIdentity("OGN", "45/298"))
        price = FixturePriceProvider().get_price(card, "EN", "ES", "trend")
        self.assertEqual(price.amount, 1.67)
        self.assertEqual(price.currency, "EUR")


if __name__ == "__main__":
    unittest.main()
