import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from riftbound_scanner.official_gallery import import_official_cards, load_official_cards


def gallery_payload() -> dict:
    return {
        "props": {
            "pageProps": {
                "page": {
                    "blades": [
                        {
                            "cards": {
                                "items": [
                                    {
                                        "id": "ogn-1-298",
                                        "collectorNumber": 1,
                                        "name": "Origins Card",
                                        "publicCode": "OGN-001/298",
                                        "set": {"value": {"id": "OGN", "label": "Origins"}},
                                        "cardImage": {"url": "https://example.test/ogn.png"},
                                    },
                                    {
                                        "id": "ogs-1-24",
                                        "collectorNumber": 1,
                                        "name": "Promo Card",
                                        "publicCode": "OGS-001/024",
                                        "set": {"value": {"id": "OGS", "label": "Proving Grounds"}},
                                        "cardImage": {"url": "https://example.test/ogs.png"},
                                    },
                                ]
                            }
                        }
                    ]
                }
            }
        }
    }


class OfficialGalleryImporterTest(unittest.TestCase):
    def test_import_official_cards_filters_sets_and_writes_aggregate_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "official_cards.json"
            with patch("riftbound_scanner.official_gallery.fetch_official_gallery", return_value=gallery_payload()):
                payload = import_official_cards(
                    output_path=output,
                    set_codes=["OGS"],
                    download_images=False,
                )

            self.assertEqual(payload["set_codes"], ["OGS"])
            self.assertEqual(payload["sets"][0]["image_directory"], "images/official/proving-grounds")
            self.assertEqual(payload["available_sets"][0]["set_code"], "OGN")
            self.assertEqual(payload["available_sets"][1]["set_code"], "OGS")
            cards = load_official_cards(output)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].set_code, "OGS")
            self.assertEqual(cards[0].printed_number, "001/024")


if __name__ == "__main__":
    unittest.main()
