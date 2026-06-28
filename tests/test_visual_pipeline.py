import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from riftbound_scanner.models import Card
from riftbound_scanner.official_gallery import (
    available_sets,
    extract_cards,
    local_image_path,
    official_card_to_dict,
)
from riftbound_scanner.vector_store import CardVectorStore
from riftbound_scanner.visual import CardPreprocessor


class VisualPipelineTest(unittest.TestCase):
    def test_official_gallery_mapping(self):
        payload = {
            "props": {
                "pageProps": {
                    "page": {
                        "blades": [
                            {
                                "cards": {
                                    "items": [
                                        {
                                            "id": "unl-1-219",
                                            "collectorNumber": 1,
                                            "name": "Arena Kingpin",
                                            "publicCode": "UNL-001/219",
                                            "set": {"value": {"id": "UNL", "label": "Unleashed"}},
                                            "cardImage": {"url": "https://example.test/card.png"},
                                        },
                                        {
                                            "id": "ogn-66a-298",
                                            "collectorNumber": 66,
                                            "name": "Ahri, Alluring",
                                            "publicCode": "OGN-066a/298",
                                            "set": {"value": {"id": "OGN", "label": "Origins"}},
                                            "cardImage": {"url": "https://example.test/ahri.png"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
        cards = extract_cards(payload, "UNL")
        mapped = official_card_to_dict(cards[0])
        self.assertEqual(mapped["card_id"], "UNL-1-219")
        self.assertEqual(mapped["printed_number"], "001/219")
        self.assertEqual(mapped["collector_number"], "001")
        self.assertEqual(mapped["image_url"], "https://example.test/card.png")
        self.assertEqual(available_sets(payload)[0]["set_code"], "OGN")

        all_cards = extract_cards(payload, None)
        self.assertEqual(len(all_cards), 2)
        origins = official_card_to_dict(extract_cards(payload, ["OGN"])[0])
        self.assertEqual(origins["printed_number"], "066a/298")
        self.assertEqual(origins["collector_number"], "066a")
        self.assertEqual(origins["pricecharting_set_slug"], "riftbound-origins")
        self.assertEqual(
            local_image_path(origins).as_posix().split("images/official/", 1)[1],
            "origins/066a_ahri_alluring.png",
        )

    def test_preprocessor_outputs_square_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "card.jpg"
            Image.new("RGB", (744, 1039), (80, 120, 160)).save(path)
            result = CardPreprocessor(output_size=384).preprocess(path, use_detector=False)
            self.assertEqual(result.image.size, (384, 384))
            self.assertTrue(result.debug_image.startswith("data:image/jpeg;base64,"))

    def test_vector_store_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CardVectorStore(temp_dir, "cards")
            card = Card(
                card_id="UNL-1",
                name="Arena Kingpin",
                set_code="UNL",
                set_name="Unleashed",
                printed_number="1/219",
                collector_number="1",
            )
            rows = [
                CardVectorStore.card_to_row(card, np.array([1.0, 0.0, 0.0], dtype="float32")),
                CardVectorStore.card_to_row(
                    Card(
                        card_id="UNL-2",
                        name="Another Card",
                        set_code="UNL",
                        set_name="Unleashed",
                        printed_number="2/219",
                        collector_number="2",
                    ),
                    np.array([0.0, 1.0, 0.0], dtype="float32"),
                ),
            ]
            store.create(rows)
            results = store.search(np.array([1.0, 0.0, 0.0], dtype="float32"), top_k=1)
            self.assertEqual(results[0]["card_id"], "UNL-1")


if __name__ == "__main__":
    unittest.main()
