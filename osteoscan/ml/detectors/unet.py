"""
unet_segmenter.py — U-Net cortical bone segmenter.

Loads model/unet_weights.keras if present.
Takes a cropped ROI (numpy BGR), returns a binary mask (H,W) uint8.
Falls back to an all-ones mask (pass-through) if model not found.

Weights NOT included — must be trained on ROI crops with cortical bone masks.
See model/README.md for training notes.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

UNET_WEIGHTS = Path(__file__).resolve().parents[3] / "model" / "unet_weights.keras"
UNET_SIZE = (256, 256)


@functools.lru_cache(maxsize=1)
def load_unet_model():
    """Load U-Net from model/unet_weights.keras. Returns None if not found."""
    if not UNET_WEIGHTS.exists():
        return None
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(UNET_WEIGHTS))
    except Exception:
        return None


def segment_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Segment cortical bone from a mandibular ROI crop.
    Returns binary mask (H,W) uint8 — 255=bone, 0=background.
    Falls back to all-255 mask if model absent.
    """
    model = load_unet_model()
    orig_h, orig_w = roi_bgr.shape[:2]

    if model is None:
        return np.full((orig_h, orig_w), 255, dtype=np.uint8)

    try:
        resized = cv2.resize(roi_bgr, UNET_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch = np.expand_dims(rgb, axis=0)
        pred = model.predict(batch, verbose=0)[0]
        if pred.ndim == 3:
            pred = pred[..., 0]
        binary = (pred > 0.5).astype(np.uint8) * 255
        return cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return np.full((orig_h, orig_w), 255, dtype=np.uint8)


def apply_mask(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to ROI, zeroing out non-bone pixels. Returns masked BGR."""
    masked = roi_bgr.copy()
    masked[mask == 0] = 0
    return masked
