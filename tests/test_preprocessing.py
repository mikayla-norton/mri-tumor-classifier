"""Tests for src.preprocessing.

Verifies the shape/dtype contract that src.inference and training/train.py
both rely on, and that the function is deterministic given the same
input (a property the training pipeline depends on for reproducibility).
"""

import numpy as np
import pytest
from PIL import Image

from src import config
from src.preprocessing import preprocess_batch, preprocess_image


def _solid_image(size=(300, 200), color=(10, 20, 30), mode="RGB") -> Image.Image:
    return Image.new(mode, size, color)


def test_output_shape_is_batch_of_one():
    batch = preprocess_image(_solid_image())
    assert batch.shape == (1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


def test_output_dtype_is_float32():
    batch = preprocess_image(_solid_image())
    assert batch.dtype == np.float32


def test_respects_custom_image_size():
    batch = preprocess_image(_solid_image(), image_size=64)
    assert batch.shape == (1, 64, 64, 3)


def test_grayscale_input_is_converted_to_three_channels():
    batch = preprocess_image(_solid_image(mode="L", color=128))
    assert batch.shape == (1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


def test_rgba_input_is_handled():
    batch = preprocess_image(_solid_image(mode="RGBA", color=(1, 2, 3, 255)))
    assert batch.shape == (1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


def test_preprocessing_is_deterministic():
    image = _solid_image()
    first = preprocess_image(image)
    second = preprocess_image(image)
    np.testing.assert_array_equal(first, second)


def test_preprocess_batch_matches_single_image_calls():
    images = [_solid_image(color=(i, i, i)) for i in (10, 20, 30)]
    batch = preprocess_batch(images)
    assert batch.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    for i, image in enumerate(images):
        np.testing.assert_array_equal(batch[i], preprocess_image(image)[0])


def test_channel_order_is_bgr_to_match_shipped_model():
    # A pure-red RGB image should come out with the highest value in the
    # *last* channel (blue-in-BGR-position-of-what-was-red) - i.e. this
    # test would fail loudly if someone "fixes" the channel order without
    # also retraining the model. See src/preprocessing.py module docstring.
    red_image = _solid_image(color=(255, 0, 0))
    batch = preprocess_image(red_image)
    pixel = batch[0, 0, 0]
    assert pixel[2] > pixel[0]  # R channel ends up last (BGR order)
