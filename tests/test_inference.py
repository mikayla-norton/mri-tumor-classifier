"""Tests for src.inference.predict.

Uses a minimal fake model (anything with a `.predict(batch)` method) so
these tests exercise the output *structure* and confidence/uncertainty
logic without requiring TensorFlow to be installed.
"""

import numpy as np
import pytest
from PIL import Image

from src import config
from src.inference import predict


class FakeModel:
    """Stands in for a Keras model: returns a fixed softmax distribution
    regardless of input, so tests can control exactly what "prediction"
    comes back."""

    def __init__(self, probabilities):
        self._probabilities = np.array(probabilities, dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.ndim == 4  # sanity check: predict() must pass a real batch
        return np.tile(self._probabilities, (batch.shape[0], 1))


def _image():
    return Image.new("RGB", (150, 150), (100, 100, 100))


def test_predict_returns_expected_keys():
    model = FakeModel([0.1, 0.1, 0.1, 0.7])
    result = predict(_image(), model)
    assert set(result.keys()) == {
        "class",
        "display_name",
        "confidence",
        "probabilities",
        "is_uncertain",
    }


def test_predict_picks_argmax_class():
    # Index 3 -> "pituitary" per config.CLASS_NAMES ordering.
    model = FakeModel([0.05, 0.05, 0.05, 0.85])
    result = predict(_image(), model)
    assert result["class"] == "pituitary"
    assert result["display_name"] == config.CLASS_DISPLAY_NAMES["pituitary"]
    assert result["confidence"] == pytest.approx(0.85, abs=1e-5)


def test_probabilities_cover_all_classes():
    model = FakeModel([0.25, 0.25, 0.25, 0.25])
    result = predict(_image(), model)
    assert set(result["probabilities"].keys()) == set(config.CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_high_confidence_is_not_flagged_uncertain():
    model = FakeModel([0.02, 0.02, 0.02, 0.94])
    result = predict(_image(), model)
    assert not result["is_uncertain"]


def test_low_confidence_is_flagged_uncertain():
    # Roughly uniform -> top class confidence well under the threshold.
    model = FakeModel([0.28, 0.27, 0.24, 0.21])
    result = predict(_image(), model)
    assert result["is_uncertain"]
    assert result["confidence"] < config.UNCERTAIN_CONFIDENCE_THRESHOLD
