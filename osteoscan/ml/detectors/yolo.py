"""
yolo_detector.py — YOLOv8 mandibular ROI detector.

Loads yolo_weights.pt from model/yolo_weights.pt if present.
Returns list of (x1,y1,x2,y2) boxes for detected mandibular cortex ROIs.
Falls back to returning the full image bbox if model not found or no detections.

Weights NOT included — must be trained on annotated panoramic X-ray dataset
with mandibular ROI bounding box labels. See model/README.md for training notes.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2

YOLO_WEIGHTS = Path(__file__).resolve().parents[3] / "model" / "yolo_weights.pt"
YOLO_CONF = 0.10


@functools.lru_cache(maxsize=1)
def load_yolo_model():
    """Load YOLOv8 model from model/yolo_weights.pt. Returns None if not found."""
    if not YOLO_WEIGHTS.exists():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(YOLO_WEIGHTS))
    except Exception:
        return None


def detect_roi(img_path: str, conf: float = YOLO_CONF) -> list[tuple]:
    """
    Detect mandibular cortex ROI boxes in a panoramic X-ray.
    Returns list of (x1,y1,x2,y2). Falls back to full image if model absent.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return [(0, 0, 1, 1)]

    h, w = img.shape[:2]
    model = load_yolo_model()

    if model is None:
        return [(0, 0, w, h)]

    try:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        boxes = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    boxes.append((x1, y1, x2, y2))
        if boxes:
            return boxes
    except Exception:
        pass

    return [(0, 0, w, h)]
