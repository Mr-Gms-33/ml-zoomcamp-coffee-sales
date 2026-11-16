"""Upload model + static Space (works on free HF accounts)."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

MODEL_ID = os.environ.get("HF_MODEL_ID", "RoxMoni/coffee-sales-classifier")
STATIC_SPACE_ID = os.environ.get("HF_STATIC_SPACE_ID", "RoxMoni/coffee-sales-midterm")
ROOT = Path(__file__).resolve().parent


def main() -> None:
    api = HfApi()

    api.create_repo(repo_id=MODEL_ID, repo_type="model", exist_ok=True)
    upload_folder(
        folder_path=ROOT / "models",
        repo_id=MODEL_ID,
        repo_type="model",
        path_in_repo=".",
    )

    card = (ROOT / "models" / "metrics.json").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=(
            f"---\n"
            f"title: Coffee Drink Classifier\n"
            f"library_name: sklearn\n"
            f"tags: [ml-zoomcamp, classification]\n"
            f"---\n\n"
            f"RandomForest pipeline for drink prediction. Metrics:\n\n"
            f"```json\n{card}\n```\n"
        ).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=MODEL_ID,
        repo_type="model",
    )

    api.create_repo(
        repo_id=STATIC_SPACE_ID,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=(ROOT / "hf_static" / "index.html").read_bytes(),
        path_in_repo="index.html",
        repo_id=STATIC_SPACE_ID,
        repo_type="space",
    )

    print(f"Model: https://huggingface.co/{MODEL_ID}")
    print(f"Space: https://huggingface.co/spaces/{STATIC_SPACE_ID}")


if __name__ == "__main__":
    main()