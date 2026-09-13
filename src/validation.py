"""Validation for user-uploaded images.

Deliberately has no TensorFlow / Streamlit dependency so it can be unit
tested quickly and reused anywhere (CLI, notebook, app).

This module answers "is this a well-formed image we can safely run
preprocessing on?" - it makes NO claim about whether the image is
actually an MRI scan. See docstrings for why that distinction matters.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from src import config


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an uploaded file.

    `image` is populated only when `ok` is True.
    """

    ok: bool
    error_message: str | None = None
    image: Image.Image | None = None


def _has_allowed_extension(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in config.ALLOWED_UPLOAD_EXTENSIONS)


def validate_upload(
    file_obj: BinaryIO,
    filename: str,
    *,
    max_size_mb: int = config.MAX_UPLOAD_SIZE_MB,
) -> ValidationResult:
    """Validate an uploaded file before it ever reaches the model.

    Checks, in order:
      1. File extension is one we claim to support.
      2. File size is under the configured maximum.
      3. The bytes actually decode as an image (guards against renamed /
         corrupt / non-image files).
      4. The decoded image meets a minimum plausible size.

    Args:
        file_obj: A file-like object opened in binary mode, positioned at
            the start (e.g. Streamlit's UploadedFile, or an io.BytesIO).
        filename: Original filename, used only for the extension check.
        max_size_mb: Maximum accepted upload size in megabytes.

    Returns:
        A ValidationResult. If ok is False, error_message explains why in
        language suitable for display directly to the user.
    """

    if not _has_allowed_extension(filename):
        allowed = ", ".join(config.ALLOWED_UPLOAD_EXTENSIONS)
        return ValidationResult(
            ok=False,
            error_message=(
                f"Unsupported file type. Please upload one of: {allowed}."
            ),
        )

    raw_bytes = file_obj.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return ValidationResult(
            ok=False,
            error_message=(
                f"File is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {max_size_mb} MB."
            ),
        )
    if len(raw_bytes) == 0:
        return ValidationResult(ok=False, error_message="The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        # .verify() confirms the file isn't truncated/corrupt, but leaves
        # the image unusable afterwards, so we re-open for actual use.
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return ValidationResult(
            ok=False,
            error_message=(
                "This file couldn't be read as an image. It may be "
                "corrupted or not actually an image file."
            ),
        )

    width, height = image.size
    if width < config.MIN_IMAGE_DIMENSION or height < config.MIN_IMAGE_DIMENSION:
        return ValidationResult(
            ok=False,
            error_message=(
                f"Image is too small ({width}x{height}px). Please upload a "
                f"scan of at least {config.MIN_IMAGE_DIMENSION}x"
                f"{config.MIN_IMAGE_DIMENSION}px."
            ),
        )

    # Normalize to RGB here so every downstream consumer (preprocessing,
    # display) can assume a 3-channel image regardless of whether the
    # source was grayscale, RGBA, or palette-based.
    if image.mode != "RGB":
        image = image.convert("RGB")

    return ValidationResult(ok=True, image=image)
