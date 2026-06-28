from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import re
import ssl
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog import CardCatalog
from .models import CardIdentity
from .pricing import default_price_provider
from .visual import VisualCardMatcher

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"
IMAGE_DIR = ROOT / "images"
ANNOTATION_DIR = ROOT / "annotations"
ANNOTATION_LABELS = ["card"]


class RiftboundHandler(BaseHTTPRequestHandler):
    catalog = CardCatalog.from_json()
    price_provider = default_price_provider()
    visual_matcher = VisualCardMatcher()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/images/"):
            root = ROOT
            relative = parsed.path.lstrip("/")
        else:
            root = PUBLIC_DIR
            relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        file_path = (root / relative).resolve()
        if not str(file_path).startswith(str(root.resolve())) or not file_path.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "visual_index_available": self.visual_matcher.available,
                    "embedding_model": self.visual_matcher.embedder.model_name,
                    "price_provider": self.price_provider.__class__.__name__,
                }
            )
            return
        if parsed.path == "/api/dataset/images":
            self._send_json(self._dataset_images())
            return
        if parsed.path.startswith("/api/annotations/"):
            image_id = parsed.path.removeprefix("/api/annotations/")
            self._send_json(self._load_annotation(image_id))
            return
        if parsed.path == "/api/export/coco":
            self._send_json(self._export_coco())
            return
        if parsed.path == "/api/cards":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._send_json({"cards": [self._card_json(card) for card in self.catalog.search(query)]})
            return
        if parsed.path.startswith("/api/cards/"):
            parts = parsed.path.removeprefix("/api/cards/").split("/", 1)
            if len(parts) != 2:
                self._send_error(HTTPStatus.BAD_REQUEST, "Expected /api/cards/{set}/{number}.")
                return
            card = self.catalog.resolve(CardIdentity(parts[0], parts[1]))
            if not card:
                self._send_error(HTTPStatus.NOT_FOUND, "Card not found.")
                return
            self._send_json({"card": self._card_json(card)})
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dataset/upload":
            try:
                result = self._upload_dataset_image()
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._send_json(result)
            return
        if parsed.path.startswith("/api/annotations/"):
            try:
                image_id = parsed.path.removeprefix("/api/annotations/")
                payload = self._read_json()
                self._save_annotation(image_id, payload)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json({"status": "saved"})
            return
        if parsed.path == "/api/scan-image":
            try:
                result = self._scan_image()
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._send_json(result)
            return
        if parsed.path == "/api/detect-region":
            try:
                result = self._detect_region()
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._send_json(result)
            return
        if parsed.path != "/api/resolve":
            self._send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        try:
            payload = self._read_json()
            result = self._resolve(payload)
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_json(result)

    def _scan_image(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        image_field = form["image"] if "image" in form else None
        if image_field is None or not getattr(image_field, "file", None):
            raise ValueError("Provide an image file.")

        language = self._field_value(form, "language", "EN").upper()
        seller_country = self._field_value(form, "seller_country", "ES").upper()
        price_mode = self._field_value(form, "price_mode", "min").lower()
        if price_mode not in {"min", "trend"}:
            raise ValueError("price_mode must be 'min' or 'trend'.")

        suffix = Path(getattr(image_field, "filename", "") or "card.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
            temp.write(image_field.file.read())
            temp.flush()
            image_path = temp.name
            top_k = int(self._field_value(form, "top_k", "5"))
            result = self.visual_matcher.search(image_path, top_k=top_k)

        matches = [self._visual_match_json(match) for match in result.matches]
        best = matches[0] if matches else None
        price = None
        if result.matches:
            price = self.price_provider.get_price(result.matches[0].card, language, seller_country, price_mode)

        return {
            "card": best["card"] if best else None,
            "best_match": best,
            "matches": matches,
            "confidence": best["score"] if best else 0.0,
            "price": self._price_json(price) if price else None,
            "debug": result.debug,
            "latency_ms": result.latency_ms,
            "input": {
                "language": language,
                "seller_country": seller_country,
                "price_mode": price_mode,
                "top_k": top_k,
            },
            "warnings": result.warnings,
        }

    def _detect_region(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        image_field = form["image"] if "image" in form else None
        if image_field is None or not getattr(image_field, "file", None):
            raise ValueError("Provide an image file.")

        suffix = Path(getattr(image_field, "filename", "") or "frame.jpg").suffix or ".jpg"
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
            temp.write(image_field.file.read())
            temp.flush()
            detections = self.visual_matcher.preprocessor.detector.detect(temp.name, confidence_threshold=0.45)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        card_detections = [item for item in detections if item.label == "card"]
        best = max(card_detections, key=lambda item: item.confidence) if card_detections else None
        return {
            "available": self.visual_matcher.preprocessor.detector.available,
            "latency_ms": latency_ms,
            "detections": [
                {**item.as_dict(), "selected": item is best}
                for item in detections
            ],
        }

    def _upload_dataset_image(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        image_field = form["image"] if "image" in form else None
        if image_field is None or not getattr(image_field, "file", None):
            raise ValueError("Provide an image file.")

        filename = Path(getattr(image_field, "filename", "") or "sample.jpg").name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Supported image formats: jpg, jpeg, png, webp.")

        target_dir = IMAGE_DIR / "user_samples"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(filename).stem).strip("._") or "sample"
        target = target_dir / f"{safe_stem}{suffix}"
        counter = 2
        while target.exists():
            target = target_dir / f"{safe_stem}_{counter}{suffix}"
            counter += 1
        target.write_bytes(image_field.file.read())

        rel = target.relative_to(ROOT).as_posix()
        return {
            "status": "uploaded",
            "image": {
                "id": self._image_id(rel),
                "path": rel,
                "url": f"/{rel}",
                "annotated": False,
            },
        }

    def _resolve(self, payload: dict[str, object]) -> dict[str, object]:
        language = str(payload.get("language") or "EN").upper()
        seller_country = str(payload.get("seller_country") or "ES").upper()
        price_mode = str(payload.get("price_mode") or "min").lower()
        if price_mode not in {"min", "trend"}:
            raise ValueError("price_mode must be 'min' or 'trend'.")

        warnings: list[str] = []
        card_id = str(payload.get("card_id") or "").strip()
        set_code = str(payload.get("set_code") or "").strip()
        printed_number = str(payload.get("printed_number") or "").strip()
        if card_id:
            card = next((item for item in self.catalog.cards if item.card_id == card_id), None)
            if not card:
                raise ValueError("Card id was not found.")
            price = self.price_provider.get_price(card, language, seller_country, price_mode)
            return {
                "card": self._card_json(card),
                "confidence": 1.0,
                "price": self._price_json(price),
                "input": {
                    "card_id": card_id,
                    "language": language,
                    "seller_country": seller_country,
                    "price_mode": price_mode,
                },
                "debug": {},
                "warnings": warnings,
            }
        if set_code and printed_number:
            identity = CardIdentity(set_code.upper(), printed_number.lower())
            confidence = 1.0
        else:
            raise ValueError("Provide either card_id or set_code + printed_number.")

        return self._resolve_identity(
            identity,
            confidence,
            language,
            seller_country,
            price_mode,
            warnings,
        )

    def _resolve_identity(
        self,
        identity,
        confidence: float,
        language: str,
        seller_country: str,
        price_mode: str,
        warnings: list[str],
        debug: dict[str, object] | None = None,
    ) -> dict[str, object]:
        card, match_confidence, match_reason = self.catalog.resolve_fuzzy(identity)
        if card and match_confidence < 1:
            confidence = min(confidence, match_confidence)
        if not card:
            warnings.append(f"No card found for {identity.set_code} {identity.printed_number}.")
            return {
                "card": None,
                "confidence": 0.0,
                "price": None,
                "input": {
                    "set_code": identity.set_code,
                    "printed_number": identity.printed_number,
                    "language": language,
                    "seller_country": seller_country,
                    "price_mode": price_mode,
                },
                "debug": debug or {},
                "warnings": warnings,
            }
        elif match_reason:
            warnings.append(match_reason)

        price = self.price_provider.get_price(card, language, seller_country, price_mode)  # type: ignore[arg-type]
        return {
            "card": self._card_json(card),
            "confidence": confidence,
            "price": self._price_json(price),
            "input": {
                "set_code": identity.set_code,
                "printed_number": identity.printed_number,
                "language": language,
                "seller_country": seller_country,
                "price_mode": price_mode,
            },
            "debug": debug or {},
            "warnings": warnings,
        }

    def _serve_static(self, path: str) -> None:
        if path.startswith("/images/"):
            self._serve_file_from_root(path, ROOT)
            return
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        self._serve_file_from_root(relative, PUBLIC_DIR)

    def _serve_file_from_root(self, path: str, root: Path) -> None:
        relative = path.lstrip("/")
        file_path = (root / relative).resolve()
        if not str(file_path).startswith(str(root.resolve())) or not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dataset_images(self) -> dict[str, object]:
        images: list[dict[str, object]] = []
        for path in sorted(IMAGE_DIR.glob("**/*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            image_id = self._image_id(rel)
            images.append(
                {
                    "id": image_id,
                    "path": rel,
                    "url": f"/{rel}",
                    "annotated": self._annotation_path(image_id).exists(),
                }
            )
        return {"labels": ANNOTATION_LABELS, "images": images}

    def _load_annotation(self, image_id: str) -> dict[str, object]:
        self._validate_image_id(image_id)
        path = self._annotation_path(image_id)
        if not path.exists():
            return {"image_id": image_id, "labels": ANNOTATION_LABELS, "boxes": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_annotation(self, image_id: str, payload: dict[str, object]) -> None:
        self._validate_image_id(image_id)
        boxes = payload.get("boxes")
        if not isinstance(boxes, list):
            raise ValueError("Annotation payload must include boxes.")
        clean_boxes = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            label = str(box.get("label") or "")
            if label not in ANNOTATION_LABELS:
                raise ValueError(f"Invalid label: {label}")
            clean_boxes.append(
                {
                    "label": label,
                    "x": float(box.get("x", 0)),
                    "y": float(box.get("y", 0)),
                    "width": float(box.get("width", 0)),
                    "height": float(box.get("height", 0)),
                }
            )
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        self._annotation_path(image_id).write_text(
            json.dumps({"image_id": image_id, "labels": ANNOTATION_LABELS, "boxes": clean_boxes}, indent=2),
            encoding="utf-8",
        )

    def _export_coco(self) -> dict[str, object]:
        images = self._dataset_images()["images"]
        categories = [{"id": index + 1, "name": label} for index, label in enumerate(ANNOTATION_LABELS)]
        category_ids = {item["name"]: item["id"] for item in categories}
        coco_images = []
        annotations = []
        annotation_id = 1
        for image_index, image in enumerate(images, start=1):
            image_id = str(image["id"])
            coco_images.append({"id": image_index, "file_name": image["path"]})
            payload = self._load_annotation(image_id)
            for box in payload.get("boxes", []):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_index,
                        "category_id": category_ids[str(box["label"])],
                        "bbox": [box["x"], box["y"], box["width"], box["height"]],
                        "area": box["width"] * box["height"],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        return {"images": coco_images, "annotations": annotations, "categories": categories}

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    @staticmethod
    def _image_id(path: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "__", path)

    @staticmethod
    def _validate_image_id(image_id: str) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", image_id):
            raise ValueError("Invalid image id.")

    @staticmethod
    def _annotation_path(image_id: str) -> Path:
        return ANNOTATION_DIR / f"{image_id}.json"

    @staticmethod
    def _field_value(form: cgi.FieldStorage, name: str, default: str) -> str:
        if name not in form:
            return default
        value = form[name]
        if isinstance(value, list):
            value = value[0]
        return str(value.value or default)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    @staticmethod
    def _card_json(card) -> dict[str, object]:
        return {
            "card_id": card.card_id,
            "name": card.name,
            "set_code": card.set_code,
            "set_name": card.set_name,
            "printed_number": card.printed_number,
            "language": card.language,
            "language_code": card.language_code,
            "cardmarket_product_id": card.cardmarket_product_id,
            "collector_number": card.collector_number,
            "pricecharting_url": card.pricecharting_url,
            "pricecharting_set_slug": card.pricecharting_set_slug,
            "pricecharting_product_id": card.pricecharting_product_id,
            "pricecharting_ungraded_usd": card.pricecharting_ungraded_usd,
            "local_image_path": card.local_image_path,
            "source_title": card.source_title,
            "image_url": card.image_url,
        }

    @staticmethod
    def _price_json(price) -> dict[str, object]:
        return {
            "mode": price.mode,
            "amount": price.amount,
            "currency": price.currency,
            "source": price.source,
            "updated_at": price.updated_at,
            "filters": price.filters,
            "message": price.message,
        }

    @staticmethod
    def _visual_match_json(match) -> dict[str, object]:
        return {
            "rank": match.rank,
            "score": match.score,
            "distance": match.distance,
            "card": RiftboundHandler._card_json(match.card),
        }

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Riftbound scanner MVP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--certfile", default=None)
    parser.add_argument("--keyfile", default=None)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RiftboundHandler)
    scheme = "http"
    if args.certfile and args.keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"Riftbound scanner server running at {scheme}://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
