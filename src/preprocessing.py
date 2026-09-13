"""Image preprocessing shared by training and inference.

Phase 3.1 of the roadmap: there must be exactly one definition of how a
raw image becomes model input, used by both `training/train.py` and
`src/inference.py`. If these ever drift apart, the model silently sees
different data at train and predict time.

IMPORTANT - channel order quirk (please read before changing this file)
-------------------------------------------------------------------------
The original notebook loaded images with OpenCV (`cv2.imread`), which
decodes to **BGR** channel order, and fed that directly into the model
with no ImageNet-style normalization (no `preprocess_input` call, no
scaling to [-1, 1] or [0, 1]) - just raw uint8 pixel values, in BGR
order, cast to float32.

The model shipped in `models/neuronet_mri_legacy.keras` was trained on
exactly that pipeline. So, to keep working correctly with the *existing*
frozen weights, this function intentionally reproduces BGR-order,
unnormalized preprocessing rather than the more conventional RGB
pipeline you might expect.

This is flagged as technical debt (see roadmap Phase 10 / README
Limitations): a future retrain should switch to RGB + proper
EfficientNet preprocessing, and this function should be updated
alongside it. Until a new model is trained and frozen, do not change
this function's channel order or normalization - doing so will silently
break predictions for the current model without raising any error.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src import config


def preprocess_image(
    image: Image.Image, *, image_size: int = config.IMAGE_SIZE
) -> np.ndarray:
    """Convert a decoded PIL image into a single-item model input batch.

    Args:
        image: A decoded PIL image (any mode; converted to RGB first for
            a well-defined starting point, then reordered to BGR to match
            the training pipeline - see module docstring).
        image_size: Target square size in pixels. Defaults to the size
            the shipped model was trained on (see config.IMAGE_SIZE).

    Returns:
        A float32 array of shape (1, image_size, image_size, 3), BGR
        channel order, pixel values in [0, 255], matching what the
        shipped model was trained on.
    """

    array = _pil_to_training_array(image, image_size=image_size)
    return np.expand_dims(array, axis=0)


def preprocess_batch(
    images: list[Image.Image], *, image_size: int = config.IMAGE_SIZE
) -> np.ndarray:
    """Preprocess many images at once (used by the training pipeline).

    Equivalent to calling preprocess_image on each image and stacking the
    results, but avoids repeated allocation overhead for large datasets.
    """

    arrays = [_pil_to_training_array(image, image_size=image_size) for image in images]
    return np.stack(arrays, axis=0)


def _pil_to_training_array(image: Image.Image, *, image_size: int) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb_array = np.asarray(image, dtype=np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(
        bgr_array, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.float32)
