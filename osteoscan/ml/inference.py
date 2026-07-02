"""
inference.py — EfficientNetB0 patch-based osteoporosis inference.

Pipeline:
  1. Read image (OpenCV)
  2. Remove black/white borders (bounding rect of non-zero pixels)
  3. Apply CLAHE on LAB L-channel (clipLimit=2.0, tileGridSize=(8,8))
  4. Split into non-overlapping 100×100 patches (discard partial)
  5. Batch: resize each to 100×100, ensure 3ch, BGR→RGB, float32 (raw [0,255])
  6. Predict all patches in one forward pass
  7. Soft-vote: average class probabilities across patches → label + probs

Classes (alphabetical, matching training): Normal=0, Osteopenia=1, Osteoporosis=2
"""

import functools

import cv2
import numpy as np

PATCH_SIZE = 100
CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


def remove_border(img: np.ndarray) -> np.ndarray:
    """Remove black/white border by finding bounding rect of non-zero pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to LAB L-channel; returns BGR."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a, b]), cv2.COLOR_LAB2BGR)


def split_into_patches(img: np.ndarray, patch_size: int = PATCH_SIZE) -> list:
    """Return all full patch_size×patch_size non-overlapping patches."""
    h, w = img.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append(img[y : y + patch_size, x : x + patch_size])
    return patches


def preprocess_patches(patches: list) -> np.ndarray:
    """Resize, ensure 3ch, BGR→RGB, float32 raw [0,255]. Shape: (N,100,100,3)."""
    out = []
    for p in patches:
        p = cv2.resize(p, (100, 100))
        if len(p.shape) == 2:
            p = np.stack([p, p, p], axis=-1)
        elif p.shape[2] == 1:
            p = np.concatenate([p, p, p], axis=-1)
        else:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        out.append(p.astype(np.float32))
    return np.array(out)


@functools.lru_cache(maxsize=1)
def load_model(model_path: str):
    """Load and cache the Keras model."""
    try:
        from tensorflow import keras

        return keras.models.load_model(model_path)
    except Exception:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)


def predict(img_path: str, model_path: str, class_names: list = None) -> dict:
    """
    Run full pipeline on one image.

    Returns:
        {'label': str, 'probs': {name: float}, 'patch_count': int}
    Raises:
        FileNotFoundError if image cannot be read.
        ValueError if no patches are extracted.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = remove_border(img)
    img = apply_clahe(img)
    patches = split_into_patches(img, patch_size=PATCH_SIZE)

    if not patches:
        img_resized = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
        patches = [img_resized]

    batch = preprocess_patches(patches)
    model = load_model(str(model_path))
    preds = model.predict(batch, verbose=0)

    avg_probs = preds.mean(axis=0)
    label_idx = int(np.argmax(avg_probs))

    return {
        "label": class_names[label_idx],
        "probs": {name: float(avg_probs[i]) for i, name in enumerate(class_names)},
        "patch_count": len(patches),
    }
