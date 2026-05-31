"""
U-Net training loop — ported from Unet_Extraction.ipynb.

Three training runs are supported:
  - "left"    : unet_roi_left.pth
  - "right"   : unet_roi_right.pth
  - "general" : unet_roi_general.pth

NOTE: One left-side training run in the source notebook diverged (loss ≈ 2.0254).
This is recorded here for completeness and must NOT be "fixed" — the training
code is ported verbatim.

Random seeds are set for reproducibility (numpy, random, torch) without altering
training behaviour.
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .unet import UNet
from .dataset import ROIDatasetPad


_SPLIT_SEED = 42


def train_unet(
    images_dir: str,
    masks_dir: str,
    side: str = "general",
    output_path: str = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    val_split: float = 0.2,
    device: str = None,
) -> str:
    """
    Train a U-Net model and save weights.

    Args:
        side: one of "left", "right", "general" — determines output filename.
        val_split: fraction for validation (80/20 random_split as in source).
                   Pass 0.0 to skip validation.
        output_path: explicit save path; if None, defaults to
                     "unet_roi_{side}.pth".

    Returns:
        Path to saved weights file.

    NOTE: left-side training diverged (loss 2.0254) in original notebook.
          This code does not fix that behaviour.
    """
    # Set seeds — does not alter training logic
    random.seed(_SPLIT_SEED)
    np.random.seed(_SPLIT_SEED)
    torch.manual_seed(_SPLIT_SEED)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_path is None:
        output_path = f"unet_roi_{side}.pth"

    dataset = ROIDatasetPad(images_dir, masks_dir)

    if val_split > 0.0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, _ = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(_SPLIT_SEED),
        )
    else:
        train_ds = dataset

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / max(1, len(loader))
        print(f"Epoch [{epoch}/{epochs}] loss: {avg:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved: {output_path}")
    return output_path
