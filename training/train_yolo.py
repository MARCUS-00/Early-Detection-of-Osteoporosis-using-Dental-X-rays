"""
train_yolo.py — Train YOLOv8 mandibular-ROI detector.

REQUIRES annotated data in YOLO format:
  data/yolo/
    images/{train,val}/*.jpg          (panoramic X-rays)
    labels/{train,val}/*.txt          (YOLO bbox: class cx cy w h, normalised)
    data.yaml                         (paths + class names)

This data does NOT exist in the current project (patches only, no panoramics,
no bounding boxes). Obtain annotated panoramic X-rays first.

Run: python train_yolo.py --epochs 100
Output: model/yolo_weights.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = _ROOT / "data" / "yolo" / "data.yaml"
OUTPUT = _ROOT / "model" / "yolo_weights.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8 mandibular ROI detector.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        print(
            "ERROR: data/yolo/data.yaml not found.\n"
            "Training requires annotated panoramic X-ray data in YOLO format:\n"
            "  data/yolo/images/{train,val}/*.jpg  — panoramic radiographs\n"
            "  data/yolo/labels/{train,val}/*.txt  — bbox per line: class cx cy w h (normalised)\n"
            "  data/yolo/data.yaml                 — YOLO dataset descriptor\n\n"
            "This data is NOT included in the repository.\n"
            "Obtain/annotate panoramic X-rays with mandibular cortex bounding boxes first."
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    OUTPUT.parent.mkdir(exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/yolo",
        name="mandible",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(str(best), str(OUTPUT))
        print(f"DONE: weights saved to {OUTPUT}")
    else:
        print("WARNING: training complete but best.pt not found at expected path.")


if __name__ == "__main__":
    main()
