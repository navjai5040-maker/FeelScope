from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models"
FILES = ("bigru_model.keras", "tokenizer.json", "labels.json", "config.json")


def main() -> None:
    repo_id = os.getenv("HF_MODEL_REPO")
    token = os.getenv("HF_TOKEN") or None
    if not repo_id:
        raise SystemExit("Set HF_MODEL_REPO, e.g. your-username/emotion-classifier-bigru.")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(DESTINATION),
        )
        print(f"Downloaded {filename} -> {path}")
    print("All four artifacts are ready in models/.")


if __name__ == "__main__":
    main()
