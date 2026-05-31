"""
Tests for split_into_patches — synthetic in-memory arrays only, no dataset files.
"""
import numpy as np
import pytest

from osteo.preprocessing.patches import split_into_patches


def solid_image(h: int, w: int) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 128


def test_exact_grid_count():
    """
    200x300 image with patch_size=100 -> 2 rows, 3 cols = 6 exact patches.
    """
    img = solid_image(200, 300)
    patches = split_into_patches(img, patch_size=100)
    assert len(patches) == 6, f"Expected 6 patches, got {len(patches)}"


def test_exact_size_only():
    """
    210x310 image with patch_size=100:
      - row-wise: floor(210/100)=2 full rows (remainder 10px row dropped)
      - col-wise: floor(310/100)=3 full cols (remainder 10px col dropped)
      -> 2 * 3 = 6 patches
    """
    img = solid_image(210, 310)
    patches = split_into_patches(img, patch_size=100)
    assert len(patches) == 6, f"Expected 6 patches (border discarded), got {len(patches)}"


def test_all_patches_correct_shape():
    """Every returned patch must be exactly (patch_size, patch_size, channels)."""
    img = solid_image(250, 350)
    patches = split_into_patches(img, patch_size=100)
    for i, p in enumerate(patches):
        assert p.shape == (100, 100, 3), (
            f"Patch {i} has shape {p.shape}, expected (100, 100, 3)"
        )


def test_smaller_than_patch_returns_empty():
    """An image smaller than patch_size in any dimension yields no patches."""
    img = solid_image(50, 50)
    patches = split_into_patches(img, patch_size=100)
    assert patches == [], f"Expected empty list, got {len(patches)} patches"


def test_exact_single_patch():
    """An image exactly patch_size x patch_size yields exactly one patch."""
    img = solid_image(100, 100)
    patches = split_into_patches(img, patch_size=100)
    assert len(patches) == 1
    assert patches[0].shape == (100, 100, 3)


def test_patch_content_matches_source():
    """Patch content must equal the corresponding region of the source image."""
    img = np.arange(300 * 300 * 3, dtype=np.uint8).reshape(300, 300, 3)
    patches = split_into_patches(img, patch_size=100)
    assert len(patches) == 9  # 3x3 grid
    np.testing.assert_array_equal(patches[0], img[0:100, 0:100])
    np.testing.assert_array_equal(patches[4], img[100:200, 100:200])
    np.testing.assert_array_equal(patches[8], img[200:300, 200:300])
