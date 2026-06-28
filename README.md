# Riftbound Visual Scanner MVP

This repo is now structured around visual card recognition instead of OCR:

- import official Riftbound card metadata and images from every set exposed by the official gallery
- preprocess a photographed card into a clean normalized image
- generate a visual embedding with SigLIP 2
- store fingerprints in a local LanceDB vector database
- search by image similarity and return ranked matches with latency metrics
- keep pricing providers backend-side and optional

## Install

```bash
uv sync
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full data, embedding, vector search, API, UI, and artifact policy.

Additional planning and publication notes:

- [`docs/PUBLICATION_CHECKLIST.md`](docs/PUBLICATION_CHECKLIST.md)
- [`docs/LINKEDIN_POST_DRAFT.md`](docs/LINKEDIN_POST_DRAFT.md)
- [`docs/ARCHITECTURE_IMAGE_GUIDE.md`](docs/ARCHITECTURE_IMAGE_GUIDE.md)

## Build the Official Visual Index

Download the official catalog from the Riftbound card gallery:

```bash
uv run python scripts/import_official_cards.py
```

By default this imports every set exposed by the gallery into `data/official_cards.json` and downloads images to `images/official/<set-slug>/`. The current official payload includes `OGN` Origins, `OGS` Proving Grounds, `SFD` Spiritforged, and `UNL` Unleashed. `OGNX` is not exposed as a `set.id` in the current gallery payload; official promos/proving-grounds cards are tracked through the set ids returned by the payload, currently `OGS`.

Create the local LanceDB vector index:

```bash
uv run python scripts/build_vector_index.py
```

For a quick smoke test, use `--limit 5` on both commands. To import a subset explicitly, pass comma-separated set codes:

```bash
uv run python scripts/import_official_cards.py --sets OGN,SFD
```

## Run

```bash
RIFTBOUND_PRICE_PROVIDER=pricecharting uv run python -m riftbound_scanner.server --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

For mobile live camera mode, browsers require HTTPS. Generate a local certificate for your LAN IP and run the server on HTTPS:

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

Open `https://<LAN_IP>:8443` on the phone and accept the local certificate warning.

Live camera mode runs YOLO on lightweight preview frames. The final `/api/scan-image` response returns the card match first; price is loaded separately through `/api/price` so slow provider calls do not block recognition.

## Search One Image

```bash
uv run python scripts/search_image.py --image path/to/card-photo.jpg --top-k 5
```

## Evaluate Retrieval

The evaluator creates augmented queries from official local images and reports precision and latency:

```bash
uv run python scripts/eval_visual_retrieval.py --samples 25
```

Output is written to:

```text
reports/visual_retrieval_eval.json
```

The external evaluator uses local PriceCharting images as cross-source queries against the official vector index. Import those images first:

```bash
uv run python scripts/import_pricecharting_catalog.py
```

Then run:

```bash
uv run python scripts/eval_pricecharting_retrieval.py --top-k 5
```

Output is written to:

```text
reports/pricecharting_retrieval_eval.json
```

## Test

```bash
uv run python -m unittest discover -s tests
```

## Cardmarket

The mobile app must not contain Cardmarket credentials. Configure credentials only on the backend:

```bash
export CARDMARKET_APP_TOKEN=...
export CARDMARKET_APP_SECRET=...
export CARDMARKET_ACCESS_TOKEN=...
export CARDMARKET_ACCESS_SECRET=...
```

The implementation supports `pricecharting` (default), `fixture`, and `cardmarket` providers.

## Region Detector Training

YOLO is now only needed to detect the full card in real photos. The current annotation tool still supports historical labels, but new training data should focus on:

- `card`

Prepare and train:

```bash
uv run python scripts/prepare_yolo_dataset.py
uv run python scripts/train_region_detector.py
uv run python scripts/export_region_detector.py
```

The exported detector is expected at:

```text
models/riftbound_regions.onnx
```

If the model is missing, the scanner falls back to an OpenCV contour crop and then full-image preprocessing.

## Universal TCG Detector Audit

The universal dataset lives in:

```text
dataset/universal_tcg_detection/
```

It uses one manifest, `annotations.jsonl`, with pixel-space `corners`, `polygon`, `bbox`, and `annotation_type` fields. `corners` and `polygon` are real localization labels. `full_image` means the image is already an isolated card and the whole image is used as the card polygon.

Prepare YOLO datasets for the two audit experiments:

```bash
uv run python scripts/prepare_universal_yolo_dataset.py --experiment localization_only
uv run python scripts/prepare_universal_yolo_dataset.py --experiment hybrid
```

Run the full audited training flow with augmentation, validation, test metrics, visual samples, and conditional ONNX export:

```bash
uv run python scripts/train_universal_region_detector.py
```

The audit output is written to:

```text
reports/region_detector_training_audit.json
reports/region_detector_samples/
```

The model is exported to `models/riftbound_regions.onnx` only when the best experiment reaches `mAP50 >= 0.75` on the test split.

Current trained detector:

```text
models/riftbound_regions.onnx
```

It was exported from `runs/detect/runs/detect/tcg_hybrid/weights/best.pt` and is loaded automatically by `riftbound_scanner.region_detector.RegionDetector`, which is used by the visual preprocessing pipeline before the OpenCV contour fallback.

Training comparison from the June 27, 2026 audit run:

| Experiment | Labels used | Test precision | Test recall | Test mAP50 | Test mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `localization_only` | corners + polygons only | 0.9957 | 1.0000 | 0.9950 | 0.9141 |
| `hybrid` | corners + polygons + isolated full-card images | 0.9992 | 1.0000 | 0.9950 | 0.9635 |

The selected `hybrid` run was stopped manually during epoch 42 after the validation curve had stabilized enough for this scanner use case. Its best validation checkpoint was epoch 40 with `mAP50=0.9942` and `mAP50-95=0.9628`.
