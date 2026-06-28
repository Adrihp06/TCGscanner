from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from download_detector import DEFAULT_FILENAME, DEFAULT_REPO_ID, sha256_file

DEFAULT_MODEL_PATH = Path("models") / DEFAULT_FILENAME


def model_card(repo_id: str, model_path: Path, sha256: str) -> str:
    size_mb = model_path.stat().st_size / (1024 * 1024)
    return f"""---
license: other
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - trading-card-games
  - object-detection
  - yolo
  - onnx
  - riftbound
---

# TCGscanner Card Detector

This repository contains the current ONNX card-boundary detector used by the TCGscanner prototype.

The model is a single-class YOLO detector. Its task is to localize the physical trading card in a camera frame or photograph. Card identification is handled separately by SigLIP 2 visual embeddings and LanceDB vector search in the application repository.

## Files

- `{DEFAULT_FILENAME}`: exported ONNX detector expected at `models/{DEFAULT_FILENAME}` by the scanner.

## Current Artifact

- Size: `{size_mb:.2f} MB`
- SHA256: `{sha256}`
- Class labels: `card`
- Default confidence threshold in the app: `0.35`

## Training Summary

The detector was trained on a universal TCG detection dataset that combines localized card examples from multiple trading card domains. The objective is to learn generic card geometry rather than the visual identity of a specific game.

The selected hybrid experiment used corners, polygons, and isolated full-card samples. The June 27, 2026 audit run reported:

| Experiment | Test precision | Test recall | Test mAP50 | Test mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| localization_only | 0.9957 | 1.0000 | 0.9950 | 0.9141 |
| hybrid | 0.9992 | 1.0000 | 0.9950 | 0.9635 |

The selected hybrid run was stopped manually during epoch 42 after the validation curve had stabilized for the scanner use case. Its best validation checkpoint was epoch 40 with `mAP50=0.9942` and `mAP50-95=0.9628`.

## Usage

```bash
uv run python scripts/download_detector.py
```

The scanner loads the downloaded model from:

```text
models/{DEFAULT_FILENAME}
```

## Limitations

- This detector only localizes the card boundary.
- It does not identify the card.
- The current dataset still needs more real-world Riftbound photographs.
- Pricing and collection features are outside this model repository.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the current ONNX detector to Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HUGGINGFACE_TOKEN or HF_TOKEN before publishing.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=args.filename,
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
    )

    sha256 = sha256_file(model_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        readme_path = Path(temp_dir) / "README.md"
        readme_path.write_text(model_card(args.repo_id, model_path, sha256), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
        )

    print(f"Published {model_path} to https://huggingface.co/{args.repo_id}")
    print(f"sha256={sha256}")


if __name__ == "__main__":
    main()
