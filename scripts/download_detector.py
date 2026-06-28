from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_REPO_ID = "Adrihp06/TCGscanner-detector"
DEFAULT_FILENAME = "riftbound_regions.onnx"
DEFAULT_OUTPUT = Path("models") / DEFAULT_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_detector(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    output: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> Path:
    output = Path(output)
    if output.exists() and not force:
        print(f"Detector already exists: {output} sha256={sha256_file(output)}")
        return output

    from huggingface_hub import hf_hub_download

    cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, output)
    print(f"Downloaded detector: {output} sha256={sha256_file(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the current YOLO ONNX card detector from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_detector(
        repo_id=args.repo_id,
        filename=args.filename,
        output=Path(args.output),
        force=args.force,
    )


if __name__ == "__main__":
    main()
