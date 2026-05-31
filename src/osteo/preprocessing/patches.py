"""
Patch splitting — ported verbatim from Complete_implementation.ipynb.
"""
import numpy as np
from typing import List


def split_into_patches(img: np.ndarray, patch_size: int = 100) -> List[np.ndarray]:
    """
    Tile an image into non-overlapping square patches.

    Only patches whose shape is exactly (patch_size, patch_size) are kept;
    border patches that are smaller are discarded.

    Source: Complete_implementation.ipynb — ported verbatim.
    """
    patches = []
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches
