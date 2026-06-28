from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .models import Card
from .region_detector import RegionDetector

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "google/siglip2-base-patch16-384"
PREPROCESS_VERSION = "card-warp-pad-v1"
CARD_ASPECT = 744 / 1039


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    used_detector: bool
    warnings: list[str]
    debug_image: str | None = None
    detector_debug: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualMatch:
    card: Card
    score: float
    distance: float
    rank: int


@dataclass(frozen=True)
class VisualSearchResult:
    matches: list[VisualMatch]
    latency_ms: dict[str, float]
    debug: dict[str, object]
    warnings: list[str]


class CardPreprocessor:
    def __init__(self, output_size: int = 384, detector: RegionDetector | None = None) -> None:
        self.output_size = output_size
        self.detector = detector or RegionDetector()

    def preprocess(self, image_path: str | Path, use_detector: bool = True) -> PreprocessResult:
        image = Image.open(image_path).convert("RGB")
        warnings: list[str] = []
        card = image
        used_detector = False
        detections_debug: list[dict[str, object]] = []
        selected_detection: int | None = None

        if use_detector:
            detections = self.detector.detect(image_path)
            detections_debug = [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "x": item.x,
                    "y": item.y,
                    "width": item.width,
                    "height": item.height,
                    "selected": False,
                }
                for item in detections
            ]
            card_detections = [item for item in detections if item.label == "card"]
            if card_detections:
                best = max(card_detections, key=lambda item: item.confidence)
                selected_detection = detections.index(best)
                detections_debug[selected_detection]["selected"] = True
                card = image.crop(
                    (
                        max(0, int(best.x)),
                        max(0, int(best.y)),
                        min(image.width, int(best.x + best.width)),
                        min(image.height, int(best.y + best.height)),
                    )
                )
                used_detector = True
            elif detections:
                warnings.append("Detector returned no card label; using full image.")

        if not used_detector:
            contour = self._opencv_card_crop(image)
            if contour is not None:
                card = contour
            else:
                warnings.append("Card contour was not found; using full image.")

        card = ImageOps.exif_transpose(card).convert("RGB")
        card = self._trim_border(card)
        card = self._pad_to_card_ratio(card)
        card = ImageOps.pad(card, (self.output_size, self.output_size), method=Image.Resampling.LANCZOS)
        card = ImageEnhance.Sharpness(card).enhance(1.05)
        return PreprocessResult(
            image=card,
            used_detector=used_detector,
            warnings=warnings,
            debug_image=self._image_data_url(card),
            detector_debug={
                "enabled": use_detector,
                "used": used_detector,
                "image_width": image.width,
                "image_height": image.height,
                "selected_index": selected_detection,
                "detections": detections_debug,
            },
        )

    def _opencv_card_crop(self, image: Image.Image) -> Image.Image | None:
        try:
            import cv2
        except ImportError:
            return None
        if not hasattr(cv2, "cvtColor") or not hasattr(cv2, "findContours"):
            return None
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:8]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            points = approx.reshape(4, 2).astype("float32")
            ordered = self._order_points(points)
            width, height = 744, 1039
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(ordered, destination)
            warped = cv2.warpPerspective(frame, matrix, (width, height))
            return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        return None

    @staticmethod
    def _order_points(points: Any) -> Any:
        rect = np.zeros((4, 2), dtype="float32")
        sums = points.sum(axis=1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        diffs = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    @staticmethod
    def _trim_border(image: Image.Image) -> Image.Image:
        width, height = image.size
        margin_x = max(1, math.floor(width * 0.015))
        margin_y = max(1, math.floor(height * 0.015))
        if width <= margin_x * 2 or height <= margin_y * 2:
            return image
        return image.crop((margin_x, margin_y, width - margin_x, height - margin_y))

    @staticmethod
    def _pad_to_card_ratio(image: Image.Image) -> Image.Image:
        width, height = image.size
        current = width / height
        if abs(current - CARD_ASPECT) < 0.02:
            return image
        if current > CARD_ASPECT:
            target_height = int(width / CARD_ASPECT)
            pad = max(0, target_height - height)
            return ImageOps.expand(image, border=(0, pad // 2, 0, pad - pad // 2), fill=(245, 247, 248))
        target_width = int(height * CARD_ASPECT)
        pad = max(0, target_width - width)
        return ImageOps.expand(image, border=(pad // 2, 0, pad - pad // 2, 0), fill=(245, 247, 248))

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ImageEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or self._best_device()
        self._processor = None
        self._model = None

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    @property
    def processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            if self.device == "mps":
                torch.set_float32_matmul_precision("high")
        return self._model

    def embed_image(self, image: Image.Image) -> np.ndarray:
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            outputs = outputs.image_embeds
        vector = outputs.detach().cpu().float().numpy()[0]
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Image embedding norm is zero.")
        return (vector / norm).astype("float32")

    @staticmethod
    def _best_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


class VisualCardMatcher:
    def __init__(
        self,
        vector_db_path: Path | str = ROOT / "data" / "vector_db",
        table_name: str = "card_embeddings",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        from .vector_store import CardVectorStore

        self.preprocessor = CardPreprocessor()
        self.embedder = ImageEmbedder(model_name=model_name)
        self.store = CardVectorStore(vector_db_path, table_name)

    @property
    def available(self) -> bool:
        return self.store.exists()

    def search(self, image_path: str | Path, top_k: int = 5) -> VisualSearchResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(image_path)
        timings["preprocess"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        vector = self.embedder.embed_image(preprocessed.image)
        timings["embedding"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        rows = self.store.search(vector, top_k=top_k)
        timings["vector_search"] = (time.perf_counter() - start) * 1000
        timings["total"] = sum(timings.values())

        matches: list[VisualMatch] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row.get("_distance", 0.0))
            score = max(0.0, 1.0 - distance / 2.0)
            matches.append(VisualMatch(card=self.store.row_to_card(row), score=score, distance=distance, rank=rank))

        return VisualSearchResult(
            matches=matches,
            latency_ms={key: round(value, 2) for key, value in timings.items()},
            debug={
                "preprocessed_image": preprocessed.debug_image,
                "used_detector": preprocessed.used_detector,
                "detector": preprocessed.detector_debug,
                "embedding_model": self.embedder.model_name,
                "preprocess_version": PREPROCESS_VERSION,
            },
            warnings=preprocessed.warnings,
        )
