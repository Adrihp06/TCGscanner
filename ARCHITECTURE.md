# Riftbound Scanner Architecture

## Overview

The scanner identifies cards by visual similarity, not OCR. The pipeline is:

1. Import official card metadata and images from the Riftbound card gallery.
2. Normalize each card image into a stable square input.
3. Generate an image embedding with SigLIP 2.
4. Store embeddings and card metadata in a local LanceDB vector table.
5. For a query image, run the same preprocessing and embedding steps, then return nearest cards by cosine distance.

The app is intentionally local-first. Large generated artifacts can be recreated with scripts and are ignored by git.

## Data Sources

`riftbound_scanner/official_gallery.py` reads the official gallery payload embedded in:

```text
https://riftbound.leagueoflegends.com/en-us/card-gallery/
```

The current official set ids found in that payload are:

- `OGN`: Origins
- `OGS`: Proving Grounds
- `SFD`: Spiritforged
- `UNL`: Unleashed

`OGNX` is not exposed as a set id in the current official payload. Available promo-like cards are represented as `OGS` Proving Grounds.

The aggregate catalog is:

```text
data/official_cards.json
```

Official images are downloaded to:

```text
images/official/<set-slug>/
```

These images are generated data and are ignored by git.

## Visual Pipeline

`riftbound_scanner/visual.py` owns preprocessing and embedding.

Preprocessing:

- load image as RGB
- use YOLO card detection if `models/riftbound_regions.onnx` exists
- fall back to OpenCV contour crop
- fall back to full image when no card boundary is found
- trim a small border
- pad to card aspect ratio
- resize/pad to `384 x 384`
- return a debug image for API/UI inspection

Embedding:

- default model: `google/siglip2-base-patch16-384`
- backend: `transformers` + `torch`
- device: CUDA, MPS, then CPU
- vector is L2-normalized before storage/search

## Vector Store

`riftbound_scanner/vector_store.py` wraps LanceDB.

Default DB:

```text
data/vector_db/
```

Default table:

```text
card_embeddings
```

Each row stores:

- card identity and set metadata
- local/source image path
- embedding model name
- embedding dimension
- preprocessing version
- normalized vector

The vector DB is generated and ignored by git.

## API and UI

`riftbound_scanner/server.py` serves static UI and JSON endpoints.

Important endpoints:

- `GET /api/health`: checks vector index availability and embedding model
- `POST /api/scan-image`: multipart image search, returns ranked visual matches
- `POST /api/detect-region`: lightweight YOLO detection for live camera frames
- `POST /api/price`: async price lookup for a resolved card
- `GET /api/cards`: catalog search
- `GET /api/cards/{set}/{number}`: manual catalog lookup
- annotation endpoints remain for YOLO card-box training

`public/index.html`, `public/app.js`, and `public/styles.css` implement the scanner UI:

- upload/capture card photo
- choose top-K
- show best match and ranked candidates
- show score, distance, price, and latency
- show preprocessed debug image

## Scripts

Import all official cards and images:

```bash
uv run python scripts/import_official_cards.py
```

Import only selected sets:

```bash
uv run python scripts/import_official_cards.py --sets OGN,SFD,UNL,OGS
```

Build the full vector index:

```bash
uv run python scripts/build_vector_index.py
```

Search one image:

```bash
uv run python scripts/search_image.py --image path/to/card.jpg --top-k 5 --repeat 3
```

Evaluate synthetic robustness from official images:

```bash
uv run python scripts/eval_visual_retrieval.py --samples 25
```

Evaluate external local PriceCharting images:

```bash
uv run python scripts/eval_pricecharting_retrieval.py --top-k 5
```

## Evaluation

Two evaluation modes are used:

- Synthetic official-image augmentations: tests robustness to rotation, brightness, blur, compression, and border/background changes.
- PriceCharting external images: tests cross-source retrieval against local non-official images under `images/pricecharting`.

Current external PriceCharting result against the 952-card official index:

- samples: `102`
- precision@1: `1.00`
- precision@3: `1.00`
- MRR: `1.00`
- p50 total latency: about `51 ms`
- p95 total latency: about `56 ms`

The first request in a process includes model loading and is much slower. Warm latency is the relevant server metric.

## Generated Artifacts Policy

Commit:

- source code
- tests
- small metadata catalogs such as `data/official_cards.json`
- documentation

Do not commit:

- `images/official/`
- `images/pricecharting/`
- `images/user_samples/`
- `data/vector_db/`
- `annotations/`
- `reports/`
- trained models and run outputs

All ignored generated artifacts can be recreated with the import, index, train, and eval scripts.
