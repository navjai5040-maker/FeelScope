from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
ARTIFACT_NAMES = (
    "bigru_model.keras",
    "tokenizer.json",
    "labels.json",
    "config.json",
)


class ArtifactError(RuntimeError):
    """Raised when model artifacts are missing, invalid, or incompatible."""


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or default)


def _resolve_artifact_paths() -> dict[str, str]:
    local_paths = {name: MODEL_DIR / name for name in ARTIFACT_NAMES}
    if all(path.is_file() for path in local_paths.values()):
        return {name: str(path) for name, path in local_paths.items()}

    repo_id = _setting("HF_MODEL_REPO")
    token = _setting("HF_TOKEN") or None
    if not repo_id:
        missing = [name for name, path in local_paths.items() if not path.is_file()]
        raise ArtifactError(
            "Missing model artifacts in models/: " + ", ".join(missing) +
            ". Either export all four files from the notebook into models/, or set "
            "HF_MODEL_REPO in Streamlit secrets to your Hugging Face model repository "
            "(for example: username/emotion-classifier-bigru)."
        )

    try:
        from huggingface_hub import hf_hub_download

        downloaded: dict[str, str] = {}
        for filename in ARTIFACT_NAMES:
            downloaded[filename] = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=token,
            )
        return downloaded
    except Exception as exc:
        raise ArtifactError(
            f"Could not download the model files from Hugging Face repository '{repo_id}'. "
            "Check that the repository exists, contains all four artifact files, and is "
            "public or that HF_TOKEN has read access. Original error: " + str(exc)
        ) from exc


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[Any, Any, list[str], dict[str, Any]]:
    """Load the trained Keras model and the matching preprocessing artifacts once."""
    paths = _resolve_artifact_paths()

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        model = tf.keras.models.load_model(paths["bigru_model.keras"], compile=False)
        with open(paths["tokenizer.json"], "r", encoding="utf-8") as file:
            tokenizer = tokenizer_from_json(file.read())
        with open(paths["labels.json"], "r", encoding="utf-8") as file:
            labels = json.load(file)
        with open(paths["config.json"], "r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(labels, list) or not labels:
            raise ValueError("labels.json must contain a non-empty JSON list.")
        labels = [str(label) for label in labels]
        if int(model.output_shape[-1]) != len(labels):
            raise ValueError(
                f"Model has {model.output_shape[-1]} output units but labels.json contains {len(labels)} labels."
            )
        required_config = ("max_length", "padding", "truncating")
        absent = [key for key in required_config if key not in config]
        if absent:
            raise ValueError("config.json is missing: " + ", ".join(absent))
        if int(config["max_length"]) <= 0:
            raise ValueError("config.json max_length must be positive.")

        return model, tokenizer, labels, config
    except Exception as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(
            "The model files were found but could not be loaded. Confirm that the "
            "model, tokenizer, labels, and preprocessing settings all came from the "
            "same trained BiGRU notebook run. Original error: " + str(exc)
        ) from exc
