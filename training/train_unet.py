"""
train_unet.py — Train U-Net cortical-bone segmenter.

REQUIRES paired data:
  data/unet/
    images/*.png    (ROI crops, same size as used during inference)
    masks/*.png     (binary cortical bone masks — same filename as images,
                    white=bone, black=background)

This data does NOT exist in the current project (no segmentation masks).
Obtain pixel-level cortical bone masks first.

Run: python train_unet.py --epochs 50
Output: model/unet_weights.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGES = _ROOT / "data" / "unet" / "images"
DATA_MASKS = _ROOT / "data" / "unet" / "masks"
OUTPUT = _ROOT / "model" / "unet_weights.keras"
UNET_SIZE = (256, 256)


def build_unet(input_shape: tuple = (256, 256, 3)):
    """Standard encoder-decoder U-Net with skip connections."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    bridge = conv_block(p3, 256)

    u1 = layers.Concatenate()([layers.UpSampling2D()(bridge), c3])
    d1 = conv_block(u1, 128)
    u2 = layers.Concatenate()([layers.UpSampling2D()(d1), c2])
    d2 = conv_block(u2, 64)
    u3 = layers.Concatenate()([layers.UpSampling2D()(d2), c1])
    d3 = conv_block(u3, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d3)
    return keras.Model(inputs, outputs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train U-Net cortical bone segmenter.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not DATA_IMAGES.exists() or not DATA_MASKS.exists():
        print(
            "ERROR: data/unet/ not found.\n"
            "Training requires paired ROI crops and cortical bone masks:\n"
            "  data/unet/images/*.png  — ROI crops (BGR or grayscale)\n"
            "  data/unet/masks/*.png   — binary cortical bone masks (same filenames)\n\n"
            "This data is NOT included in the repository.\n"
            "Obtain/annotate ROI crops with pixel-level cortical bone masks first."
        )
        sys.exit(1)

    import cv2
    import numpy as np

    image_paths = sorted(DATA_IMAGES.glob("*.png"))
    if not image_paths:
        print("ERROR: No images found in data/unet/images/")
        sys.exit(1)

    def load_pair(ip: Path, mp: Path):
        img = cv2.resize(cv2.imread(str(ip)), UNET_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), UNET_SIZE)
        msk = (msk > 127).astype(np.float32)[..., None]
        return img, msk

    pairs = [
        load_pair(ip, DATA_MASKS / ip.name) for ip in image_paths if (DATA_MASKS / ip.name).exists()
    ]
    if not pairs:
        print("ERROR: No matching mask files found in data/unet/masks/")
        sys.exit(1)

    X = np.array([p[0] for p in pairs])
    Y = np.array([p[1] for p in pairs])
    print(f"Loaded {len(X)} image/mask pairs. Training...")

    model = build_unet((UNET_SIZE[0], UNET_SIZE[1], 3))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, Y, epochs=args.epochs, batch_size=args.batch, validation_split=0.15)

    OUTPUT.parent.mkdir(exist_ok=True)
    model.save(str(OUTPUT))
    print(f"DONE: weights saved to {OUTPUT}")


if __name__ == "__main__":
    main()
