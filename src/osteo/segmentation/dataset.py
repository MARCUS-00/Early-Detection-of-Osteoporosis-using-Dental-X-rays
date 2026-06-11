'''
Dataset classes for U-Net training — ported from Unet_Extraction.ipynb.
DISCREPANCY: ROIDataset (no Pad) is never defined in sources.
Only ROIDatasetPad is implemented. Use ROIDatasetPad everywhere.
'''
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_to_fixed_size_center(
    img: np.ndarray,
    target_h: int = 100,
    target_w: int = 100,
    value: int = 0,
) -> np.ndarray:
    '''Centre-pad image to (target_h, target_w). Does NOT crop if larger.
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    h, w = img.shape[:2]
    pad_top    = max(0, (target_h - h) // 2)
    pad_bottom = max(0, target_h - h - pad_top)
    pad_left   = max(0, (target_w - w) // 2)
    pad_right  = max(0, target_w - w - pad_left)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                               cv2.BORDER_CONSTANT, value=value)


class ROIDatasetPad(Dataset):
    '''
    PyTorch dataset: colour images + grayscale masks, centre-padded to
    (target_h, target_w). image tensor float32 (3,H,W) in [0,1].
    mask tensor float32 (1,H,W) in [0,1].
    Source: Unet_Extraction.ipynb — ported verbatim.
    '''
    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, images_dir: str, masks_dir: str,
                 target_h: int = 100, target_w: int = 100) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.target_h   = target_h
        self.target_w   = target_w
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name
        img  = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {mask_path}')
        img  = pad_to_fixed_size_center(img,  self.target_h, self.target_w, 0)
        mask = pad_to_fixed_size_center(mask, self.target_h, self.target_w, 0)
        img_tensor  = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return img_tensor, mask_tensor
