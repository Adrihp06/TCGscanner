# TCGscanner

Visual card recognition for trading card games, with Riftbound as the first evaluation domain.

The project is a reproducible scanner prototype: it detects a physical card from a mobile/web camera frame, normalizes the card crop, represents it as a visual embedding, and searches a local vector index for the closest card in the reference catalog.

The current implementation intentionally avoids OCR as the primary recognition mechanism. OCR can be useful as an auxiliary signal, but card photos are often sleeved, tilted, partially lit, affected by glare, printed in different languages, or captured with motion blur. This repository treats recognition as a visual retrieval problem instead:

1. YOLO detects the card boundary.
2. The card crop is normalized to a stable image.
3. SigLIP 2 converts the normalized image into a visual embedding.
4. LanceDB retrieves nearest neighbors from the local card index.
5. Pricing is resolved separately so recognition stays fast.

## Architecture

The system separates detection from identification:

- **Object detection:** a single-class YOLO model detects `card` regions in camera frames.
- **Visual retrieval:** `google/siglip2-base-patch16-384` maps card images into an embedding space.
- **Vector database:** LanceDB stores reference embeddings and metadata locally.
- **Runtime scanner:** the UI streams lightweight preview frames for live detection and sends a higher-quality capture for final recognition.
- **Auditability:** detector version, embedding model, preprocessing version, index version, latency, and recognition logs are tracked in the pipeline.

A visual embedding is a numerical representation of an image in a high-dimensional vector space. The model is not asked to output a card name directly. Instead, visually similar images should produce nearby vectors. That makes card identification a nearest-neighbor search problem against a curated reference catalog.

More detail:

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`docs/LINKEDIN_POST_DRAFT.md`](docs/LINKEDIN_POST_DRAFT.md)
- [`docs/ARCHITECTURE_IMAGE_GUIDE.md`](docs/ARCHITECTURE_IMAGE_GUIDE.md)
- [`docs/PUBLICATION_CHECKLIST.md`](docs/PUBLICATION_CHECKLIST.md)

## Requirements

- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)
- Network access for the first setup run
- Optional: Apple Silicon MPS, CUDA, or CPU for embedding inference

The YOLO detector is not committed to git. It is downloaded from Hugging Face during setup:

```text
https://huggingface.co/Adrihp06/TCGscanner-detector
```

## Quickstart

Install dependencies:

```bash
uv sync
```

Prepare a full local setup:

```bash
uv run python scripts/setup_project.py
```

That command:

1. downloads the current YOLO ONNX detector to `models/riftbound_regions.onnx`;
2. imports official Riftbound metadata and card images;
3. builds the local LanceDB vector index under `data/vector_db/`.

For a fast smoke test:

```bash
uv run python scripts/setup_project.py --limit 5
```

Run the scanner:

```bash
RIFTBOUND_PRICE_PROVIDER=pricecharting uv run python -m riftbound_scanner.server --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Mobile Camera Mode

Mobile browsers require HTTPS for camera access. Generate a local certificate for your LAN IP:

```bash
LAN_IP="$(ipconfig getifaddr en0)"  # replace with your phone-reachable LAN IP if needed
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
  -keyout certs/local-key.pem \
  -out certs/local-cert.pem \
  -subj "/CN=${LAN_IP}" \
  -addext "subjectAltName=IP:${LAN_IP},DNS:localhost,IP:127.0.0.1"
uv run python -m riftbound_scanner.server --host 0.0.0.0 --port 8443 \
  --certfile certs/local-cert.pem \
  --keyfile certs/local-key.pem
```

Open this from the phone and accept the local certificate warning:

```text
https://<LAN_IP>:8443
```

Live camera mode sends compressed preview frames for YOLO detection. Once the card boundary is stable, the UI sends a higher-quality capture through the full visual search pipeline.

## Manual Setup Steps

Download or refresh the detector:

```bash
uv run python scripts/download_detector.py
```

Import the official catalog:

```bash
uv run python scripts/import_official_cards.py
```

Import selected sets only:

```bash
uv run python scripts/import_official_cards.py --sets OGN,SFD,UNL,OGS
```

Build the local vector index:

```bash
uv run python scripts/build_vector_index.py
```

Search one image from the CLI:

```bash
uv run python scanner.py path/to/card-photo.jpg --top-k 5
```

## API Surface

The backend is served by `riftbound_scanner.server`.

Important endpoints:

- `GET /api/health`: checks vector index and embedding model configuration.
- `POST /api/detect-region`: lightweight YOLO detection for live camera frames.
- `POST /api/scan-image`: full visual recognition from an uploaded/captured image.
- `POST /api/price`: asynchronous price lookup for a resolved card.
- `GET /api/cards`: catalog search.
- `GET /api/cards/{set}/{number}`: manual card lookup.

## Evaluation

Run unit tests:

```bash
uv run python -m unittest discover -s tests
```

Evaluate visual retrieval with augmented official images:

```bash
uv run python scripts/eval_visual_retrieval.py --samples 25
```

Import local PriceCharting images and evaluate cross-source retrieval:

```bash
uv run python scripts/import_pricecharting_catalog.py
uv run python scripts/eval_pricecharting_retrieval.py --top-k 5
```

Current external PriceCharting result against the 952-card official index:

| Metric | Value |
| --- | ---: |
| Samples | 102 |
| Precision@1 | 1.00 |
| Precision@3 | 1.00 |
| MRR | 1.00 |
| p50 warm total latency | ~51 ms |
| p95 warm total latency | ~56 ms |

The first request in a process includes model loading and is slower. Warm latency is the relevant server metric.

## Detector Training

The current detector is a single-class YOLO model:

```text
card
```

The universal detector dataset uses one manifest:

```text
dataset/universal_tcg_detection/annotations.jsonl
```

Each entry stores pixel-space `corners`, `polygon`, `bbox`, and `annotation_type` fields. `corners` and `polygon` are real localization labels. `full_image` means the source is already an isolated card and the full image is used as the polygon.

Prepare YOLO datasets:

```bash
uv run python scripts/prepare_universal_yolo_dataset.py --experiment localization_only
uv run python scripts/prepare_universal_yolo_dataset.py --experiment hybrid
```

Train, evaluate, and export conditionally:

```bash
uv run python scripts/train_universal_region_detector.py
```

The selected `hybrid` run was stopped manually during epoch 42 after the validation curve had stabilized for this scanner use case. Its best validation checkpoint was epoch 40 with `mAP50=0.9942` and `mAP50-95=0.9628`.

| Experiment | Labels used | Test precision | Test recall | Test mAP50 | Test mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `localization_only` | corners + polygons only | 0.9957 | 1.0000 | 0.9950 | 0.9141 |
| `hybrid` | corners + polygons + isolated full-card images | 0.9992 | 1.0000 | 0.9950 | 0.9635 |

## Publishing the Detector

Publishing requires a Hugging Face token in the environment:

```bash
export HUGGINGFACE_TOKEN=...
uv run python scripts/publish_detector_to_hf.py
```

The script creates or updates:

```text
Adrihp06/TCGscanner-detector
```

It uploads:

- `riftbound_regions.onnx`
- a Hugging Face model card with SHA256, metrics, usage, and limitations

## Generated Artifacts Policy

Commit:

- source code
- tests
- small metadata catalogs
- documentation

Do not commit:

- `models/`
- `dataset/`
- `runs/`
- `images/official/`
- `images/pricecharting/`
- `images/user_samples/`
- `data/vector_db/`
- `data/huggingface/`
- `annotations/`
- `reports/`
- `artifacts/`
- local certificates or credentials

All ignored generated artifacts can be recreated with the setup, import, index, train, and evaluation scripts.

## Limitations

- The current detector needs more real-world Riftbound photos.
- Official artwork and isolated card images are not enough to fully represent mobile capture conditions.
- The current product layer is an MVP: user accounts, collection vaults, historical prices, and exchange workflows are future work.
- Pricing providers are optional backend integrations and should never be called from a client with embedded credentials.
