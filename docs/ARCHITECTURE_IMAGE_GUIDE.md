# Architecture Image Generation Guide

Use this guide to generate the LinkedIn architecture image with an AI image tool or to build it manually in Figma/Excalidraw.

## Goal

Create a clear technical architecture diagram for a TCG visual scanner. The image should communicate the complete pipeline:

- dataset ingestion;
- detector training;
- model registry;
- reference catalog embedding generation;
- vector database indexing;
- live camera detection;
- final recognition;
- async price enrichment;
- auditability/versioning.

The image must look like a serious engineering/research diagram, not a marketing hero image.

## Format

- Aspect ratio: `16:9`.
- Best size: `1920 x 1080`.
- Background: clean off-white or very light gray.
- Style: precise, scientific, minimal, high contrast.
- Typography: modern sans-serif, medium weight, readable labels.
- Avoid: 3D objects, decorative gradients, fantasy card art, neon cyberpunk, cluttered arrows, fake UI screenshots, tiny unreadable text.

## Recommended Layout

Use a left-to-right architecture with three horizontal lanes:

1. **Offline Training and Indexing**
2. **Online Live Scanner**
3. **Audit and Observability**

### Lane 1: Offline Training and Indexing

Blocks:

- Multi-TCG Dataset Sources
- Dataset Cleaning
- Annotation Conversion
- Data Augmentation
- YOLO Training
- Evaluation Metrics
- Model Registry
- Reference Card Catalog
- SigLIP 2 Embedding Generation
- LanceDB Vector Index

Flow:

`Dataset Sources -> Cleaning -> YOLO Format -> Augmentation -> YOLO Training -> Evaluation -> Model Registry`

Separate but connected reference flow:

`Reference Card Catalog -> Preprocess 384x384 -> SigLIP 2 Embeddings -> LanceDB Vector Index`

### Lane 2: Online Live Scanner

Blocks:

- Mobile/Web Camera
- Lightweight Detection Frames
- YOLO Detector
- Bounding Box Overlay
- Stable Detection Trigger
- High-Quality Capture
- Crop and Normalize
- SigLIP 2 Embedding
- Vector Search
- Ranked Card Match
- Async Price Lookup
- UI Result

Flow:

`Camera -> Compressed Preview Frame -> YOLO Detector -> Bounding Box Overlay`

Then:

`Stable Detection -> High-Quality Capture -> Crop/Normalize -> SigLIP 2 -> LanceDB Search -> Ranked Match -> UI`

Pricing side branch:

`Ranked Match -> Async Price Provider -> UI Price Enrichment`

### Lane 3: Audit and Observability

Blocks:

- Dataset Version
- Detector Version
- Embedding Model Version
- Preprocessing Config
- Vector Index Version
- Latency Metrics
- Recognition Logs

These should appear as a bottom rail connected to both offline and online lanes.

## Visual Encoding

Use consistent colors by system responsibility:

- Data and datasets: muted blue.
- Training/model lifecycle: muted purple.
- Embeddings/vector search: muted green.
- Live scanner/runtime: muted orange.
- Audit/observability: neutral gray.

Use simple icons only if they improve readability:

- camera icon for live input;
- database cylinder for LanceDB;
- chip/model icon for YOLO and SigLIP 2;
- chart icon for evaluation metrics;
- registry/archive icon for model registry.

## Exact Labels To Include

Use these labels verbatim where possible:

- Universal TCG Dataset
- Cleaning and Filtering
- YOLO Annotation Format
- Data Augmentation
- YOLO Card Detector
- Model Registry
- Riftbound Reference Catalog
- SigLIP 2 Visual Embeddings
- LanceDB Vector Index
- Live Camera Frames
- Bounding Box Overlay
- Stable Detection Trigger
- High-Quality Capture
- Crop and Normalize
- Nearest-Neighbor Search
- Ranked Card Match
- Async Price Lookup
- Dataset Version
- Model Version
- Index Version
- Latency Metrics

## Prompt

```text
Create a clean 16:9 technical architecture diagram for a trading card visual scanner system. Use a serious scientific engineering style, off-white background, sharp readable labels, minimal icons, no decorative 3D, no fantasy artwork, no fake screenshots.

The diagram has three horizontal lanes:

1. Offline Training and Indexing.
Show: Universal TCG Dataset -> Cleaning and Filtering -> YOLO Annotation Format -> Data Augmentation -> YOLO Card Detector Training -> Evaluation Metrics -> Model Registry.
Also show: Riftbound Reference Catalog -> Preprocess 384x384 -> SigLIP 2 Visual Embeddings -> LanceDB Vector Index.

2. Online Live Scanner.
Show: Mobile/Web Camera -> Live Camera Frames -> YOLO Card Detector -> Bounding Box Overlay -> Stable Detection Trigger -> High-Quality Capture -> Crop and Normalize -> SigLIP 2 Visual Embedding -> Nearest-Neighbor Search in LanceDB -> Ranked Card Match -> UI Result.
Add a side branch from Ranked Card Match to Async Price Lookup and back to UI Result.

3. Audit and Observability.
Show a bottom rail with Dataset Version, Model Version, Embedding Model Version, Preprocessing Config, Index Version, Latency Metrics, and Recognition Logs connected to the training and runtime systems.

Use muted blue for data, muted purple for model training, muted green for embeddings/vector database, muted orange for live runtime, and neutral gray for auditability. Keep all labels large and legible. Make it suitable for a LinkedIn technical post.
```

## Negative Prompt

```text
Do not include fantasy characters, collectible card artwork, neon cyberpunk styling, dark backgrounds, 3D rendered objects, illegible tiny labels, decorative gradient blobs, stock-photo people, fake mobile app screens, or overly complex arrow crossings.
```

## Manual Editing Checklist

After generation, verify:

- all labels are spelled correctly;
- LanceDB appears only as the vector database, not as the transactional user database;
- YOLO is shown as detection, not identification;
- SigLIP 2 is shown as embedding generation, not object detection;
- pricing is asynchronous and separate from recognition;
- model registry/versioning is visible;
- the diagram still reads correctly at LinkedIn feed size.
