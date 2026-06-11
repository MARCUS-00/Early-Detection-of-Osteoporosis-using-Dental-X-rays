'''
ROI extraction using a trained YOLOv8 model — ported from ROI_Extraction.ipynb.
'''
from pathlib import Path
from typing import Tuple
import numpy as np


def roi_extraction(
    img_path: str,
    model_path: str,
    conf: float = 0.1,
) -> Tuple[list, list]:
    '''
    Run YOLO inference and split detected boxes into left / right cortex ROIs.
    For each side the box with highest confidence nearest the midline is selected:
      - left side:  box with maximum x[0] (closest to center from the left)
      - right side: box with minimum x[0] (closest to center from the right)
    Returns: (left_box, right_box) as [x1, y1, x2, y2] lists.
    Raises ValueError if no detection passes confidence on either side.
    Source: ROI_Extraction.ipynb — logic ported verbatim plus empty-side guard.
    '''
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    results = model.predict(str(img_path), conf=conf)

    img = cv2.imread(str(img_path))
    img_width = img.shape[1]

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    keep = scores >= conf
    boxes = boxes[keep]

    left_boxes  = [b for b in boxes if b[0] < img_width / 2]
    right_boxes = [b for b in boxes if b[0] >= img_width / 2]

    if not left_boxes:
        raise ValueError(
            f'roi_extraction: no detections on LEFT side of {img_path!r} '
            f'(conf>={conf}). Cannot select left cortex ROI.'
        )
    if not right_boxes:
        raise ValueError(
            f'roi_extraction: no detections on RIGHT side of {img_path!r} '
            f'(conf>={conf}). Cannot select right cortex ROI.'
        )

    left_box  = max(left_boxes,  key=lambda b: b[0])
    right_box = min(right_boxes, key=lambda b: b[0])
    return left_box.tolist(), right_box.tolist()
