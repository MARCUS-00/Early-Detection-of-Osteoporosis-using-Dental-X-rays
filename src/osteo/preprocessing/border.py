"""
Border removal — ported verbatim from Complete_implementation.ipynb.
"""
import cv2
import numpy as np


def remove_border(img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Crop the black border from an image using a binary threshold.

    Converts to grayscale, thresholds at `threshold`, finds the bounding
    rectangle of nonzero pixels, and returns the crop.  Returns `img`
    unchanged if no nonzero pixels are found.

    Source: Complete_implementation.ipynb — ported verbatim.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]
