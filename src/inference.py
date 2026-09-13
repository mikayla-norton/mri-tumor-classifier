"""Single, structured prediction entry point (Phase 3.2 / 3.3).

Replaces the old pattern of calling both `model.predict(...)` and the
deprecated `model.predict_generator(...)` directly inside a Streamlit
page, and of using an ad-hoc `p < 0.95` check as a stand-in for "this
might not even be an MRI".

`predict()` takes a decoded image and a loaded model and returns a plain
dict - no Streamlit, no I/O - so it's trivial to unit test (see
tests/test_inference.py) and reusable from a CLI or batch script.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src import config
from src.preprocessing import preprocess_image


def predict(image: Image.Image, model: Any) -> dict:
    """Run the model on a single decoded image and structure the result.

    Args:
        image: A decoded, validated PIL image (see src.validation).
        model: A loaded Keras model exposing `.predict(batch)`.

    Returns:
        {
            "class": str,                 # canonical class name, e.g. "glioma"
            "display_name": str,           # human-readable label
            "confidence": float,           # softmax probability of top class
            "probabilities": dict[str, float],  # full distribution
            "is_uncertain": bool,          # confidence below configured threshold
        }

    Note on "confidence": this is the raw softmax output of the model's
    top class. It reflects the model's training distribution, not a
    calibrated real-world probability of being correct, and it is
    absolutely not a clinical confidence score. See
    config.UNCERTAIN_CONFIDENCE_THRESHOLD and the Methodology page for
    more detail.
    """

    batch = preprocess_image(image)
    raw_probabilities = model.predict(batch, verbose=0)[0]

    probabilities = {
        class_name: float(raw_probabilities[i])
        for i, class_name in enumerate(config.CLASS_NAMES)
    }

    top_index = int(np.argmax(raw_probabilities))
    top_class = config.CLASS_NAMES[top_index]
    top_confidence = float(raw_probabilities[top_index])

    return {
        "class": top_class,
        "display_name": config.CLASS_DISPLAY_NAMES[top_class],
        "confidence": top_confidence,
        "probabilities": probabilities,
        "is_uncertain": top_confidence < config.UNCERTAIN_CONFIDENCE_THRESHOLD,
    }
