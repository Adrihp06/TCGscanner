import os
import unittest
from unittest import mock

from riftbound_scanner.server import IMAGE_DIR, MAX_TOP_K, PUBLIC_DIR, RiftboundHandler


class ServerSecurityTest(unittest.TestCase):
    def test_public_static_path_blocks_traversal(self):
        self.assertIsNone(RiftboundHandler._resolve_static_path("../README.md", PUBLIC_DIR))

    def test_image_static_path_is_limited_to_image_root(self):
        self.assertIsNone(RiftboundHandler._resolve_static_path("../certs/local-key.pem", IMAGE_DIR))

    def test_image_suffix_rejects_non_image_extensions(self):
        with self.assertRaises(ValueError):
            RiftboundHandler._image_suffix("payload.txt")

    def test_top_k_is_bounded(self):
        self.assertEqual(RiftboundHandler._parse_top_k("1"), 1)
        self.assertEqual(RiftboundHandler._parse_top_k(str(MAX_TOP_K)), MAX_TOP_K)
        with self.assertRaises(ValueError):
            RiftboundHandler._parse_top_k("0")
        with self.assertRaises(ValueError):
            RiftboundHandler._parse_top_k(str(MAX_TOP_K + 1))

    def test_dataset_tools_are_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(RiftboundHandler._dataset_tools_enabled(mock.Mock()))

    def test_dataset_tools_can_be_enabled_explicitly(self):
        with mock.patch.dict(os.environ, {"RIFTBOUND_ENABLE_DATASET_TOOLS": "1"}):
            self.assertTrue(RiftboundHandler._dataset_tools_enabled(mock.Mock()))


if __name__ == "__main__":
    unittest.main()
