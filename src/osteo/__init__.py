"""
osteo — Early Detection of Osteoporosis using Dental X-rays.

Four independent stages:
  1. Preprocessing  – border removal, patch splitting
  2. ROI detection  – YOLOv8 cortex-roi detection
  3. Segmentation   – classical-CV pseudo-masks + U-Net
  4. Classification – MobileNetV2 patch-majority-vote

NOTE: The end-to-end YOLO→U-Net→classifier pipeline is NOT implemented.
      The actual deployed inference path is predict_patches.py (no YOLO, no U-Net).
"""
