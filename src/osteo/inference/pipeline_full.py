# NOT VERIFIED FROM AVAILABLE FILES
"""
YOLO → U-Net → Classifier end-to-end pipeline — PLACEHOLDER ONLY.

This wiring does NOT exist anywhere in the source code. The source files
contain three independent stages (YOLO training, U-Net training, patch
classification) with no code connecting them into a single pipeline.

The ACTUAL deployed inference path is predict_patches.py, which uses
patch-based majority voting with NO YOLO and NO U-Net.

All functions in this file raise NotImplementedError.
"""
import numpy as np


def run_full_pipeline(
    img_path: str,
    yolo_model_path: str,
    unet_model_path: str,
    classifier_model_path: str,
    class_names: list = None,
) -> str:
    # NOT VERIFIED FROM AVAILABLE FILES
    """
    Run the full YOLO→U-Net→classifier pipeline on a single image.

    NOT IMPLEMENTED — this pipeline was never assembled in the source code.
    Use osteo.inference.predict_patches.predict_image() for the verified
    inference path.
    """
    raise NotImplementedError(
        "The YOLO→U-Net→classifier end-to-end pipeline is not implemented. "
        "Use osteo.inference.predict_patches.predict_image() instead."
    )
