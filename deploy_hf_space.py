"""Upload this project to a Hugging Face Space (run after `hf auth login`)."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

SPACE_ID = os.environ.get("HF_SPACE_ID", "RoxMoni/coffee-sales-midterm")
ROOT = Path(__file__).resolve().parent


def main() -> None:
    api = HfApi()
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        exist_ok=True,
    )
    readme_hf = (ROOT / "README_HF.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=readme_hf.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
    )
    upload_folder(
        folder_path=ROOT,
        repo_id=SPACE_ID,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            "venv/*",
            "__pycache__/*",
            ".ipynb_checkpoints/*",
            "notebook.ipynb",
            "deploy_hf_space.py",
        ],
    )
    print(f"https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()