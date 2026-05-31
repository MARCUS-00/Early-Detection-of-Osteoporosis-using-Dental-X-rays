# NOT VERIFIED FROM AVAILABLE FILES
"""
ViT (vit_b_16) segmentation experiment — ABANDONED / BROKEN.

The source notebook (Unet_Extraction.ipynb) contains an incomplete and
non-functional attempt at using torchvision's vit_b_16 for segmentation.
The experiment was never completed and produced no usable results.

This file is a PLACEHOLDER ONLY per inventory rule 3.
All functions raise NotImplementedError.
"""
import numpy as np


def train_vit_segmentation(
    images_dir: str,
    masks_dir: str,
    output_path: str = "vit_segmentation.pth",
    **kwargs,
) -> str:
    # NOT VERIFIED FROM AVAILABLE FILES
    """
    Train a ViT-based segmentation model.

    NOT IMPLEMENTED — the ViT experiment in Unet_Extraction.ipynb was broken
    and abandoned before producing working code.
    """
    raise NotImplementedError(
        "ViT segmentation experiment was abandoned in the source notebook. "
        "No verified implementation exists."
    )


def predict_vit_segmentation(img: np.ndarray, model_path: str) -> np.ndarray:
    # NOT VERIFIED FROM AVAILABLE FILES
    """NOT IMPLEMENTED — see module docstring."""
    raise NotImplementedError(
        "ViT segmentation experiment was abandoned in the source notebook. "
        "No verified implementation exists."
    )
