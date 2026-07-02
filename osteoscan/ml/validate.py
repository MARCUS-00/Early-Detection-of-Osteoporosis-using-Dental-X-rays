"""
validate.py — Dental X-ray upload heuristic validator.

These are HEURISTIC checks that block obvious non-X-rays (colour photographs,
blank images, screenshots, near-black/white frames). They are NOT a trained
dental-X-ray classifier. They cannot confirm an image IS a dental X-ray; they
only reject images that clearly are NOT one. A trained ROI detector (YOLO) would
be required for true structural validation, but that data is not available here.

Checks run cheapest/most-certain first:
  1. READABLE     — cv2.imread succeeds
  2. DIMENSIONS   — w >= 100, h >= 100
  3. NOT COLOUR   — BGR channel spread (max pairwise mean diff) <= 18
  4. NOT BLANK    — grayscale std dev > 3
  5. NOT BLACK/WHITE — mean brightness in [12, 243]
  6. TONAL RANGE  — (p98 - p2) percentile spread >= 18
  7. NOT PHOTO    — HSV mean saturation <= 35

Returns (is_valid: bool, reason: str).
"""

import cv2
import numpy as np


def validate_dental_xray(img_path: str) -> tuple:
    """
    Heuristic validator — returns (True, '') or (False, user-facing reason).
    Cheap checks first; returns on first failure.
    """
    # 1. Readable
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "Could not read the file. Please upload a valid JPG or PNG image."

    h, w = img.shape[:2]

    # 2. Minimum dimensions — model patch size is 100x100, so anything >= 100x100 is valid
    if w < 100 or h < 100:
        return False, (
            f"Image is too small ({w}x{h} px). Please upload an image of at least 100x100 pixels."
        )

    # 3. Not colour — channel spread
    if len(img.shape) == 3 and img.shape[2] == 3:
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        channel_spread = max(
            abs(r.mean() - g.mean()),
            abs(r.mean() - b.mean()),
            abs(g.mean() - b.mean()),
        )
        if channel_spread > 18:
            return False, (
                "This looks like a colour photo, not a dental X-ray. "
                "Please upload a grayscale radiograph."
            )

    # Convert to grayscale for remaining checks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # 4. Not blank / low-detail
    # Real 100x100 trabecular-bone patches have std 5.8–8.7; threshold set to 3 so that
    # truly blank/near-uniform images (std ≈ 0–2) are rejected without hitting bone patches.
    if gray.std() <= 3:
        return False, ("Image looks blank or lacks detail. Please upload a real dental X-ray.")

    # 5. Not near-uniform-dark or near-uniform-bright
    mean_brightness = gray.mean()
    if mean_brightness < 12:
        return False, "Image is almost entirely black and does not look like an X-ray."
    if mean_brightness > 243:
        return False, "Image is almost entirely white and does not look like an X-ray."

    # 6. Tonal range — X-rays span a meaningful grey range.
    # Real 100x100 bone patches have p98-p2 range 24–36 (Mendeley dataset, 150 samples).
    # Threshold of 18 rejects near-uniform images while accepting all measured real patches.
    p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
    if (p98 - p2) < 18:
        return False, (
            "Image does not have the tonal range expected of an X-ray. "
            "Please upload a full-contrast radiograph."
        )

    # 7. Not obviously a colour photo by saturation
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].astype(np.float32).mean()
        if mean_sat > 35:
            return False, (
                "Image appears to be a natural colour photograph, not a radiograph. "
                "Please upload a grayscale dental X-ray."
            )

    return True, ""


def detect_panoramic_warning(img_path: str) -> str:
    """
    Heuristic check for full panoramic X-rays (not hard-reject, just a warning).

    Panoramics are typically very wide (ratio > 1.8) AND have large dark vertical
    flanks on both far edges (airway/background regions flanking the dental arch).
    Returns a user-facing warning string when both signals fire, else empty string.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return ""

    h, w = img.shape[:2]
    if h == 0:
        return ""

    # Signal 1: panoramics are distinctly wide
    if w / h <= 1.8:
        return ""

    # Signal 2: dark vertical flanks on both far edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    flank_w = max(1, w // 7)  # outer ~14 % of width on each side
    left_mean = float(gray[:, :flank_w].mean())
    right_mean = float(gray[:, -flank_w:].mean())

    if left_mean < 60 and right_mean < 60:
        return (
            "This looks like it may be a full panoramic X-ray. "
            "This model was trained on close-up periapical bone patches and "
            "may not produce reliable results on panoramic images."
        )
    return ""
