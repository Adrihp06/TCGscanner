import unittest

from riftbound_scanner.universal_yolo import (
    UniversalRecord,
    bbox_from_points,
    filter_records,
    yolo_line_from_corners,
)


class UniversalYoloTest(unittest.TestCase):
    def test_bbox_from_points(self):
        self.assertEqual(
            bbox_from_points([[10, 20], [30, 15], [40, 70], [5, 60]]),
            [5.0, 15.0, 35.0, 55.0],
        )

    def test_yolo_line_from_corners(self):
        line = yolo_line_from_corners([[0, 0], [100, 0], [100, 200], [0, 200]], 200, 400)
        self.assertEqual(line, "0 0.25000000 0.25000000 0.50000000 0.50000000")

    def test_filter_records(self):
        records = [
            UniversalRecord("a", "s", "mtg", "a.jpg", 100, 100, "corners", [[0, 0], [1, 0], [1, 1], [0, 1]]),
            UniversalRecord("b", "s", "pokemon", "b.jpg", 100, 100, "polygon", [[0, 0], [1, 0], [1, 1], [0, 1]]),
            UniversalRecord("c", "s", "ga", "c.jpg", 100, 100, "full_image", [[0, 0], [1, 0], [1, 1], [0, 1]]),
        ]
        self.assertEqual([item.id for item in filter_records(records, "localization_only")], ["a", "b"])
        self.assertEqual([item.id for item in filter_records(records, "hybrid")], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
