from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REQUIRED = ("bigru_model.keras", "tokenizer.json", "labels.json", "config.json")


def main() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    repo_id = os.getenv("HF_MODEL_REPO")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face write token) before running this script.")
    if not repo_id:
        raise SystemExit("Set HF_MODEL_REPO, e.g. your-username/emotion-classifier-bigru.")

    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    if missing:
        raise SystemExit("Missing files in models/: " + ", ".join(missing) + ". Export them from the notebook first.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(REQUIRED),
        commit_message="Upload trained FeelScope BiGRU and preprocessing artifacts",
    )
    print(f"Uploaded model artifacts to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
