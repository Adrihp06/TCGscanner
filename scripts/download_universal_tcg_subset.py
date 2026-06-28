from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "dataset" / "universal_tcg_detection"

MTG_SEG_REPO = "dhvazquez/mtg_synthetic_cards_semantic_segmentation"
POKEMON_ANNOTATED_REPO = "netprtony/pokemon-cards-image-and-annotations"
POKEMON_SEG_REPO = "antokun/pokemon-card-segmentation"
POKEMON_URL_REPO = "tooni/pokemoncards"
GRAND_ARCHIVE_REPO = "acidtib/tcg-ga-cards"
MIXED_TCG_REPO = "heftyTuna/tcg-cards-dataset"

DEFAULT_QUOTAS = {
    "riftbound_local": 0,
    "mtg_segmentation": 300,
    "pokemon_annotated": 100,
    "pokemon_segmentation": 100,
    "pokemon_url": 300,
    "grand_archive": 800,
    "mixed_tcg": 400,
}


def bbox_from_corners(corners: list[list[float]]) -> list[float]:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def full_image_corners(width: int, height: int) -> list[list[float]]:
    return [
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ]


def record(
    *,
    source: str,
    tcg: str,
    sample_id: str,
    image_path: Path,
    output: Path,
    width: int,
    height: int,
    corners: list[list[float]],
    annotation_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{tcg}/{sample_id}",
        "source": source,
        "source_tcg": tcg,
        "image_path": image_path.relative_to(output).as_posix(),
        "width": width,
        "height": height,
        "objects": [
            {
                "label": "card",
                "polygon": corners,
                "corners": corners,
                "bbox": bbox_from_corners(corners),
                "annotation_type": annotation_type,
            }
        ],
        "metadata": metadata or {},
    }


def save_pil(image: Image.Image, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, quality=95)
    return image.size


def copy_image(source: Path, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with Image.open(destination) as image:
        return image.size


def safe_sample_id(path_or_name: str) -> str:
    return Path(path_or_name).stem.replace("/", "__").replace(" ", "_")


def local_riftbound(output: Path, limit: int) -> list[dict[str, Any]]:
    candidates = sorted(
        path
        for root in [ROOT / "images" / "official", ROOT / "images" / "pricecharting"]
        if root.exists()
        for path in root.glob("**/*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    records: list[dict[str, Any]] = []
    for source_path in candidates[:limit]:
        rel = source_path.relative_to(ROOT / "images").as_posix()
        destination = output / "images" / "riftbound" / rel
        width, height = copy_image(source_path, destination)
        corners = full_image_corners(width, height)
        records.append(
            record(
                source="local:images",
                tcg="riftbound",
                sample_id=safe_sample_id(rel),
                image_path=destination,
                output=output,
                width=width,
                height=height,
                corners=corners,
                annotation_type="full_image",
                metadata={"source_relative_path": rel},
            )
        )
    return records


def mtg_segmentation(output: Path, limit: int) -> list[dict[str, Any]]:
    annotation_path = Path(
        hf_hub_download(MTG_SEG_REPO, "corner_annotations.json", repo_type="dataset")
    )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        for filename, raw_corners in sorted(annotations.get(split, {}).items()):
            if len(records) >= limit:
                return records
            source_path = Path(
                hf_hub_download(MTG_SEG_REPO, f"{split}/images/{filename}", repo_type="dataset")
            )
            destination = output / "images" / "mtg" / filename
            width, height = copy_image(source_path, destination)
            corners = [[float(x), float(y)] for x, y in raw_corners]
            records.append(
                record(
                    source=MTG_SEG_REPO,
                    tcg="mtg",
                    sample_id=safe_sample_id(filename),
                    image_path=destination,
                    output=output,
                    width=width,
                    height=height,
                    corners=corners,
                    annotation_type="corners",
                    metadata={"split": split},
                )
            )
    return records


def pokemon_annotated(output: Path, limit: int) -> list[dict[str, Any]]:
    info = HfApi().dataset_info(POKEMON_ANNOTATED_REPO)
    filenames = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("test/images/")
        and sibling.rfilename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    records: list[dict[str, Any]] = []
    for repo_path in filenames[:limit]:
        source_path = Path(hf_hub_download(POKEMON_ANNOTATED_REPO, repo_path, repo_type="dataset"))
        destination = output / "images" / "pokemon" / Path(repo_path).name
        width, height = copy_image(source_path, destination)
        corners = full_image_corners(width, height)
        records.append(
            record(
                source=POKEMON_ANNOTATED_REPO,
                tcg="pokemon",
                sample_id=safe_sample_id(repo_path),
                image_path=destination,
                output=output,
                width=width,
                height=height,
                corners=corners,
                annotation_type="full_image",
                metadata={"repo_path": repo_path},
            )
        )
    return records


def yolo_polygon_to_corners(values: list[float], width: int, height: int) -> list[list[float]]:
    coords = values[1:]
    points = []
    for index in range(0, len(coords), 2):
        points.append([coords[index] * width, coords[index + 1] * height])
    return points


def pokemon_segmentation(output: Path, limit: int) -> list[dict[str, Any]]:
    tar_path = Path(hf_hub_download(POKEMON_SEG_REPO, "343182_pokemon.tar", repo_type="dataset"))
    records: list[dict[str, Any]] = []
    with tarfile.open(tar_path) as archive:
        names = sorted(
            name
            for name in archive.getnames()
            if name.startswith("images/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for image_name in names:
            if len(records) >= limit:
                break
            label_name = image_name.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
            try:
                image_member = archive.extractfile(image_name)
                label_member = archive.extractfile(label_name)
            except KeyError:
                continue
            if image_member is None or label_member is None:
                continue
            image = Image.open(io.BytesIO(image_member.read())).convert("RGB")
            label_line = label_member.read().decode("utf-8").strip().splitlines()[0]
            values = [float(value) for value in label_line.split()]
            destination = output / "images" / "pokemon_segmentation" / Path(image_name).name
            width, height = save_pil(image, destination)
            corners = yolo_polygon_to_corners(values, width, height)
            records.append(
                record(
                    source=POKEMON_SEG_REPO,
                    tcg="pokemon",
                    sample_id=safe_sample_id(image_name),
                    image_path=destination,
                    output=output,
                    width=width,
                    height=height,
                    corners=corners,
                    annotation_type="polygon",
                    metadata={"repo_path": image_name},
                )
            )
    return records


def image_from_url(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": "TCGscaner/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def pokemon_url(output: Path, limit: int) -> list[dict[str, Any]]:
    dataset = load_dataset(POKEMON_URL_REPO, split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for row in dataset:
        if len(records) >= limit:
            break
        url = str(row.get("large_image_source") or row.get("small_image_source") or "").strip('"')
        if not url:
            continue
        try:
            image = image_from_url(url)
        except Exception:
            continue
        sample_id = safe_sample_id(str(row.get("id") or len(records)))
        destination = output / "images" / "pokemon_url" / f"{sample_id}.jpg"
        width, height = save_pil(image, destination)
        corners = full_image_corners(width, height)
        records.append(
            record(
                source=POKEMON_URL_REPO,
                tcg="pokemon",
                sample_id=sample_id,
                image_path=destination,
                output=output,
                width=width,
                height=height,
                corners=corners,
                annotation_type="full_image",
                metadata={"name": row.get("name"), "set": row.get("set_name"), "url": url},
            )
        )
    return records


def parquet_image_dataset(
    output: Path,
    *,
    repo: str,
    tcg: str,
    subdir: str,
    limit: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset(repo, split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if len(records) >= limit:
            break
        image = row.get("image")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        sample_id = safe_sample_id(f"{subdir}_{index:06d}")
        destination = output / "images" / subdir / f"{sample_id}.jpg"
        width, height = save_pil(image, destination)
        corners = full_image_corners(width, height)
        metadata: dict[str, Any] = {}
        if row.get("text") is not None:
            metadata["text"] = str(row["text"])[:1000]
        if row.get("messages") is not None:
            metadata["messages"] = str(row["messages"])[:1000]
        records.append(
            record(
                source=repo,
                tcg=tcg,
                sample_id=sample_id,
                image_path=destination,
                output=output,
                width=width,
                height=height,
                corners=corners,
                annotation_type="full_image",
                metadata=metadata,
            )
        )
    return records


def grand_archive(output: Path, limit: int) -> list[dict[str, Any]]:
    return parquet_image_dataset(
        output,
        repo=GRAND_ARCHIVE_REPO,
        tcg="grand_archive",
        subdir="grand_archive",
        limit=limit,
    )


def mixed_tcg(output: Path, limit: int) -> list[dict[str, Any]]:
    return parquet_image_dataset(
        output,
        repo=MIXED_TCG_REPO,
        tcg="mixed_tcg",
        subdir="mixed_tcg",
        limit=limit,
    )


SOURCE_BUILDERS = {
    "riftbound_local": local_riftbound,
    "mtg_segmentation": mtg_segmentation,
    "pokemon_annotated": pokemon_annotated,
    "pokemon_segmentation": pokemon_segmentation,
    "pokemon_url": pokemon_url,
    "grand_archive": grand_archive,
    "mixed_tcg": mixed_tcg,
}


def scaled_quotas(total: int) -> dict[str, int]:
    base_total = sum(DEFAULT_QUOTAS.values())
    quotas = {
        name: int(total * amount / base_total)
        for name, amount in DEFAULT_QUOTAS.items()
    }
    remainder = total - sum(quotas.values())
    for name in DEFAULT_QUOTAS:
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    return quotas


def write_manifest(output: Path, records: list[dict[str, Any]], quotas: dict[str, int]) -> None:
    manifest = output / "annotations.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    counts_by_tcg: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    counts_by_annotation_type: dict[str, int] = {}
    for item in records:
        counts_by_tcg[item["source_tcg"]] = counts_by_tcg.get(item["source_tcg"], 0) + 1
        counts_by_source[item["source"]] = counts_by_source.get(item["source"], 0) + 1
        annotation_type = item["objects"][0]["annotation_type"]
        counts_by_annotation_type[annotation_type] = counts_by_annotation_type.get(annotation_type, 0) + 1

    summary = {
        "schema": "universal_tcg_detection.v1",
        "annotation_units": "pixels",
        "bbox_format": "xywh_pixels",
        "object_fields": ["label", "polygon", "corners", "bbox", "annotation_type"],
        "notes": [
            "corners and polygon annotations are real localization labels.",
            "full_image annotations mean the image is already an isolated card, so the whole image is used as the card polygon.",
            "riftbound_local is excluded from default quotas because official gallery images are not representative detector photos.",
        ],
        "total": len(records),
        "requested_quotas": quotas,
        "counts_by_annotation_type": counts_by_annotation_type,
        "counts_by_tcg": counts_by_tcg,
        "counts_by_source": counts_by_source,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_dataset(output: Path, quotas: dict[str, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, quota in quotas.items():
        if quota <= 0:
            continue
        print(f"{name}: target {quota}", flush=True)
        source_records = SOURCE_BUILDERS[name](output, quota)
        print(f"{name}: wrote {len(source_records)}", flush=True)
        records.extend(source_records)
    return records


def parse_source_quota(values: Iterable[str]) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    for value in values:
        name, raw_amount = value.split("=", 1)
        if name not in SOURCE_BUILDERS:
            raise SystemExit(f"Unknown source {name!r}. Options: {', '.join(SOURCE_BUILDERS)}")
        quotas[name] = int(raw_amount)
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a universal multi-TCG card detection subset from local and HF sources."
    )
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--source-quota", action="append", default=[], metavar="SOURCE=N")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)

    quotas = parse_source_quota(args.source_quota) if args.source_quota else scaled_quotas(args.total)
    records = build_dataset(output, quotas)
    write_manifest(output, records, quotas)

    print(f"Wrote {len(records)} records to {output / 'annotations.jsonl'}")
    print(f"Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
