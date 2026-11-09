from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_many(
    model: Any,
    tokenizer: Any,
    labels: list[str],
    config: dict,
    texts: list[str],
) -> list[dict]:
    """Run the notebook-matched text -> sequence -> padding -> BiGRU pipeline."""
    clean_texts = [str(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(clean_texts)
    padded = pad_sequences(
        sequences,
        maxlen=int(config["max_length"]),
        padding=str(config["padding"]),
        truncating=str(config["truncating"]),
    )
    raw_predictions = model.predict(padded, verbose=0)
    probabilities = np.asarray(raw_predictions, dtype=np.float64)

    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError(
            f"Unexpected model output shape {probabilities.shape}; expected (batch, {len(labels)})."
        )

    results = []
    for row in probabilities:
        predicted_index = int(np.argmax(row))
        results.append({
            "label": labels[predicted_index],
            "confidence": float(row[predicted_index]),
            "probabilities": [float(value) for value in row],
        })
    return results
