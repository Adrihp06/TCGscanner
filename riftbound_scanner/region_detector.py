from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ["card"]


@dataclass(frozen=True)
class RegionDetection:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


class RegionDetector:
    def __init__(self, model_path: str | Path = "models/riftbound_regions.onnx") -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(self, image_path: str | Path, confidence_threshold: float = 0.35) -> list[RegionDetection]:
        if not self.available:
            return []
        results = self.model.predict(str(image_path), conf=confidence_threshold, verbose=False)
        detections: list[RegionDetection] = []
        for result in results:
            names: dict[int, str] = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                detections.append(
                    RegionDetection(
                        label=names.get(class_id, LABELS[class_id] if class_id < len(LABELS) else str(class_id)),
                        confidence=float(box.conf[0].item()),
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                    )
                )
        return detections

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path), task="detect")
        return self._model
