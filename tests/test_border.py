"""
Tests for remove_border — synthetic in-memory arrays only, no dataset files.
"""
import numpy as np
import pytest

from osteo.preprocessing.border import remove_border


def make_bgr(h: int, w: int, fill: int = 200) -> np.ndarray:
    """Return a solid-colour BGR uint8 array."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_all_black_returns_unchanged():
    """When the image is entirely black (all zeros), coords is None -> unchanged."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = remove_border(img)
    assert result is img, "Expected the same object back when no nonzero pixels"


def test_no_border_image_unchanged():
    """
    A uniformly bright image has no border; the crop should cover the full extent.
    The bounding rect of all pixels equals the full image, so shape is preserved.
    """
    img = make_bgr(80, 120, fill=150)
    result = remove_border(img)
    assert result.shape == img.shape


def test_border_is_cropped():
    """
    Image with a black border around a bright centre — the border should be removed.

    Layout:
      10-pixel black border on all sides around a 60x80 bright interior.
      After remove_border the result shape should be the interior size (60, 80, 3).
    """
    H, W = 80, 100
    BORDER = 10
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Bright interior
    img[BORDER:H - BORDER, BORDER:W - BORDER] = 200

    result = remove_border(img)

    expected_h = H - 2 * BORDER
    expected_w = W - 2 * BORDER
    assert result.shape[0] == expected_h, (
        f"Height after crop: {result.shape[0]}, expected {expected_h}"
    )
    assert result.shape[1] == expected_w, (
        f"Width after crop: {result.shape[1]}, expected {expected_w}"
    )


def test_asymmetric_border():
    """Black border only on top/left — result should remove only those borders."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[15:, 20:] = 180  # bright region starts at row 15, col 20

    result = remove_border(img)

    # Crop should start at (row=15, col=20) and end at image edge
    assert result.shape[0] == 60 - 15
    assert result.shape[1] == 80 - 20


def test_threshold_default_is_5():
    """Pixels with value exactly 5 should be treated as background (not content)."""
    img = np.full((50, 50, 3), 5, dtype=np.uint8)
    # A single bright pixel in the interior
    img[25, 25] = 100
    result = remove_border(img)
    assert result.shape[0] == 1 and result.shape[1] == 1
