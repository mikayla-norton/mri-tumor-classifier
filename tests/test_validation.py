"""Tests for src.validation - upload validation, invalid image handling.

These tests intentionally avoid TensorFlow entirely so they can run in
any environment with just PIL/numpy installed.
"""

import io

import pytest
from PIL import Image

from src.validation import validate_upload


def _jpeg_bytes(size=(200, 200), mode="RGB") -> io.BytesIO:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_valid_jpeg_is_accepted():
    result = validate_upload(_jpeg_bytes(), "scan.jpg")
    assert result.ok
    assert result.image is not None
    assert result.image.mode == "RGB"


def test_valid_png_is_accepted():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, format="PNG")
    buf.seek(0)
    result = validate_upload(buf, "scan.png")
    assert result.ok


def test_grayscale_image_is_converted_to_rgb():
    result = validate_upload(_jpeg_bytes(mode="L"), "scan.jpg")
    assert result.ok
    assert result.image.mode == "RGB"


def test_disallowed_extension_is_rejected():
    result = validate_upload(io.BytesIO(b"irrelevant"), "scan.gif")
    assert not result.ok
    assert "Unsupported file type" in result.error_message


def test_corrupt_bytes_are_rejected():
    result = validate_upload(io.BytesIO(b"this is not an image"), "scan.jpg")
    assert not result.ok
    assert "couldn't be read" in result.error_message


def test_empty_file_is_rejected():
    result = validate_upload(io.BytesIO(b""), "scan.jpg")
    assert not result.ok


def test_undersized_image_is_rejected():
    result = validate_upload(_jpeg_bytes(size=(10, 10)), "scan.jpg")
    assert not result.ok
    assert "too small" in result.error_message


def test_oversized_file_is_rejected():
    huge = io.BytesIO(b"0" * (11 * 1024 * 1024))
    result = validate_upload(huge, "scan.jpg", max_size_mb=10)
    assert not result.ok
    assert "too large" in result.error_message


def test_max_size_is_configurable():
    small_file = _jpeg_bytes()
    size_mb = len(small_file.getvalue()) / (1024 * 1024)
    result = validate_upload(small_file, "scan.jpg", max_size_mb=size_mb / 2)
    assert not result.ok
