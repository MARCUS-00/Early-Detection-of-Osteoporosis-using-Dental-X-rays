"""
Minimal I/O helpers used across the pipeline.
"""
from pathlib import Path
import cv2
import numpy as np


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image in BGR format via OpenCV. Raises FileNotFoundError if missing."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_image_gray(path: str | Path) -> np.ndarray:
    """Read an image as grayscale via OpenCV. Raises FileNotFoundError if missing."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
