import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.download_detector import download_detector, sha256_file


class SetupScriptsTest(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(b"tcgscanner")
            self.assertEqual(
                sha256_file(path),
                "6e896dab97fe2cb3e4dbbc62327f70f71c0247034953289785a44b9240ccb4d5",
            )

    def test_download_detector_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "riftbound_regions.onnx"
            output.write_bytes(b"already-here")
            with mock.patch("huggingface_hub.hf_hub_download") as mocked_download:
                result = download_detector(output=output)
            self.assertEqual(result, output)
            mocked_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
