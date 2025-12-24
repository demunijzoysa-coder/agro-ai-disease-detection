from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

CLASS_NAMES = ["blast", "healthy"]


@dataclass
class Prediction:
    image_path: str
    predicted: str
    prob_blast: float
    prob_healthy: float
    threshold: float


_model_cache: Dict[str, tf.keras.Model] = {}


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load and cache a Keras model for reuse (fast in Streamlit)."""
    p = Path(model_path)
    key = str(p.resolve())
    if key in _model_cache:
        return _model_cache[key]
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    model = tf.keras.models.load_model(p)
    _model_cache[key] = model
    return model


def _preprocess_image(image_path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    threshold: float = 0.5,
) -> Prediction:
    """
    Predict blast vs healthy.

    IMPORTANT:
    With label_mode="binary" and class_names ['blast','healthy'],
    sigmoid output is P(class==1) => P(healthy).
    Therefore:
      prob_healthy = sigmoid_output
      prob_blast   = 1 - prob_healthy
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    x = _preprocess_image(p)

    prob_healthy = float(model.predict(x, verbose=0)[0][0])
    prob_blast = 1.0 - prob_healthy

    predicted = "healthy" if prob_healthy >= threshold else "blast"

    return Prediction(
        image_path=str(p),
        predicted=predicted,
        prob_blast=prob_blast,
        prob_healthy=prob_healthy,
        threshold=threshold,
    )
