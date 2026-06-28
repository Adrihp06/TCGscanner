# LinkedIn Post Draft

I have been developing a visual recognition pipeline for trading card games, with Riftbound as the first evaluation domain.

The objective is not merely to classify an image, but to build a reproducible scanner architecture: a system that can detect a physical card from a mobile camera, normalize the visual evidence, compare it against a reference catalog, and return a ranked identification with enough latency discipline to support a live user experience.

The central design decision was to avoid treating the problem as OCR-first recognition.

OCR is useful, but it is fragile as the primary mechanism in this context. Cards may be sleeved, tilted, partially occluded, affected by glare, printed in different languages, or captured under motion blur. The more robust formulation is visual retrieval: detect the card, represent it as a visual embedding, and compare it against a curated reference index.

The architecture is organized around two independent model responsibilities:

1. Object detection: a YOLO model detects the physical card boundary.
2. Visual retrieval: SigLIP 2 maps the normalized card crop into an embedding space.

A visual embedding is a numerical representation of an image in a high-dimensional vector space. The model is not asked to directly output a card name. Instead, it produces a vector that captures visual structure: artwork, layout, color distribution, borders, typography, and other image-level signals. If two images are visually and semantically close, their vectors should also be close under a similarity metric.

This changes the recognition problem from direct classification to nearest-neighbor search.

The reference pipeline is generated offline:

- collect card metadata and official reference images;
- normalize each reference image to a consistent model input;
- compute one SigLIP 2 embedding per reference card;
- store embeddings, card identifiers, set metadata, language, and image provenance in LanceDB;
- rebuild the vector index as the catalog evolves.

The scanner pipeline runs online:

- receive a camera frame or uploaded image;
- run YOLO to detect the card region;
- crop and normalize the detected card;
- compute the SigLIP 2 embedding for the normalized crop;
- query LanceDB for nearest visual neighbors;
- return a ranked match with metadata;
- resolve pricing separately so recognition is not blocked by external providers.

For the detector, I trained a single-class YOLO model (`card`) on a universal TCG dataset. The intent was deliberate: the detector should learn the geometry of a trading card, not the identity of a specific game. The dataset combines examples from multiple TCG domains, including MTG, Pokemon, Grand Archive, and other card sources. This reduces the risk of overfitting the detection model to one artwork style, one border design, or one catalog.

The training process is treated as a versioned artifact pipeline:

- source datasets are downloaded and normalized into a common schema;
- annotations are converted into YOLO format;
- non-representative samples are removed to reduce dataset noise;
- augmentation is applied to simulate mobile capture conditions;
- model weights, dataset version, training parameters, and evaluation outputs are recorded;
- the selected detector is promoted into the scanner through a lightweight model registry convention.

This model registry step is important for auditability. A scanner result should be attributable to a specific detector version, embedding model, preprocessing configuration, and reference index. Without that discipline, it becomes difficult to explain regressions when a new model, dataset, or compression strategy changes recognition quality.

LanceDB was selected for the vector database layer because the prototype needed fast local retrieval, simple reproducibility, and low operational overhead. At this stage, the scientific question was not whether a distributed vector database could be operated at scale. The first question was whether visual embeddings plus careful preprocessing could identify real card photos reliably enough to justify the product direction.

The live camera mode validates the end-to-end interaction. The browser sends lightweight frames to the backend, YOLO returns the card bounding box, and the UI overlays the detection on the video stream. Once the detection is stable, the system captures a higher-quality frame and runs the full recognition pipeline. This separation is necessary: the detection loop can be optimized for frequency, while the recognition path must preserve enough image detail for accurate embedding comparison.

A notable performance finding was that the embedding step was not the only bottleneck. Warm embedding inference was already in the tens of milliseconds in my environment. The larger practical gains came from system design:

- keeping models loaded in memory;
- compressing only live detection frames;
- preserving quality for final recognition crops;
- decoupling visual recognition from price lookups;
- returning the match first and enriching it asynchronously.

The prototype currently integrates PriceCharting as an initial pricing source. The broader product architecture will need user accounts, per-TCG collections, set-level vaults, acquisition cost tracking, historical prices, portfolio metrics, and eventually exchange-oriented marketplace workflows.

The current limitation is also clear: the detector needs more real-world Riftbound photographs, not only official artwork or synthetic examples. That is the next data-quality problem to solve.

Even as an MVP, the result supports the core hypothesis: a practical TCG scanner can be built by combining object detection, visual embeddings, vector search, and careful pipeline versioning, without relying on OCR as the primary recognition strategy.

I will include an architecture diagram and a short demo video showing the live detection overlay and the recognition result.
