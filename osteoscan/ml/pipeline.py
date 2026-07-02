"""
pipeline.py — Full 3-stage osteoporosis inference.

Stage 1: yolo_detector.detect_roi()   -> mandibular ROI box(es)
                                          (fallback: full image if no weights)
Stage 2: unet_segmenter.segment_roi() -> cortical bone mask per ROI
                                          (fallback: pass-through if no weights)
Stage 3: inference.predict on the masked ROI patches -> Normal/Osteopenia/Osteoporosis

Returns the same dict shape as inference.predict PLUS stage metadata so the UI
can show which stages ran on real models vs fallback.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from . import inference as _inference
from .detectors import unet as unet_segmenter
from .detectors import yolo as yolo_detector


def run_pipeline(img_path: str, model_path: str) -> dict:
    """
    Run full 3-stage pipeline on one image.

    Returns:
        {
          'label':      str,
          'probs':      {class_name: float},
          'patch_count': int,
          'stages':     {'yolo': 'model'|'fallback', 'unet': 'model'|'fallback'},
          'roi_count':  int,
        }
    """
    yolo_status = "model" if yolo_detector.load_yolo_model() is not None else "fallback"
    unet_status = "model" if unet_segmenter.load_unet_model() is not None else "fallback"

    # Stage 1: detect ROIs
    boxes = yolo_detector.detect_roi(img_path)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:
        raise FileNotFoundError(f"Model weights not found or invalid: {model_path}")

    # Stage 2 + 3: segment + classify each ROI
    roi_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_bgr = img[y1:y2, x1:x2]

            # Stage 2: cortical bone segmentation (or pass-through)
            mask = unet_segmenter.segment_roi(roi_bgr)
            masked_roi = unet_segmenter.apply_mask(roi_bgr, mask)

            # Stage 3: EfficientNet on masked ROI
            roi_path = os.path.join(tmpdir, f"roi_{i}.png")
            cv2.imwrite(roi_path, masked_roi)
            try:
                result = _inference.predict(roi_path, str(model_path))
                roi_results.append(result)
            except Exception:
                pass

    if not roi_results:
        # Fallback: run directly on the original image
        roi_results = [_inference.predict(img_path, str(model_path))]

    # Average probabilities across all ROIs (soft vote)
    class_names = _inference.CLASS_NAMES
    avg_probs = {
        cls: sum(r["probs"][cls] for r in roi_results) / len(roi_results) for cls in class_names
    }
    label_idx = int(np.argmax([avg_probs[c] for c in class_names]))

    return {
        "label": class_names[label_idx],
        "probs": avg_probs,
        "patch_count": sum(r["patch_count"] for r in roi_results),
        "stages": {"yolo": yolo_status, "unet": unet_status},
        "roi_count": len(roi_results),
    }
