"""
Crop ROI patches from YOLO label files — ported from Unet_Extraction.ipynb.
"""
from pathlib import Path
import cv2
import numpy as np


def crop_from_labels(
    img_path: str,
    label_path: str,
    output_dir: str,
) -> None:
    """
    Read a YOLO-format label file, denormalize boxes, and save crops.

    Side assignment:
      - "left"  if normalised x-centre < 0.5
      - "right" otherwise

    Source: Unet_Extraction.ipynb — ported verbatim.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    stem = Path(img_path).stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, x_c, y_c, bw, bh = (float(p) for p in parts[:5])

        # Denormalise
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        side = "left" if x_c < 0.5 else "right"
        crop = img[max(0, y1):y2, max(0, x1):x2]
        save_path = out / f"{stem}_{side}_{idx}.png"
        cv2.imwrite(str(save_path), crop)
