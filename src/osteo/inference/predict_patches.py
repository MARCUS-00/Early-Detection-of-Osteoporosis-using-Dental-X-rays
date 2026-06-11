import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from osteo.preprocessing.border import remove_border
from osteo.preprocessing.patches import split_into_patches

IMG_SIZE = (100, 100)


def _preprocess_patches(patches):
    result = []
    for p in patches:
        p = cv2.resize(p, IMG_SIZE)
        if len(p.shape) == 2:
            p = np.stack([p, p, p], axis=-1)
        elif p.shape[2] == 1:
            p = np.concatenate([p, p, p], axis=-1)
        else:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        result.append(p.astype(np.float32))
    return np.array(result)


def predict_image(
    img_path: str,
    model_path: str,
    class_names: list = None,
    patch_size: int = 100,
) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    img = remove_border(img)
    patches = split_into_patches(img, patch_size=patch_size)
    if not patches:
        raise ValueError(f'No patches extracted from: {img_path}')

    from osteo.utils.compat import load_model_compat
    model = load_model_compat(model_path)

    batch = _preprocess_patches(patches)
    preds = model.predict(batch, verbose=0)

    indices = list(np.argmax(preds, axis=1))
    majority_idx, _ = Counter(indices).most_common(1)[0]

    if class_names is not None:
        return class_names[majority_idx]
    return str(majority_idx)


def predict_image_with_probs(
    img_path: str,
    model_path: str,
    class_names: list = None,
    patch_size: int = 100,
) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    img = remove_border(img)
    patches = split_into_patches(img, patch_size=patch_size)
    if not patches:
        raise ValueError(f'No patches extracted from: {img_path}')

    from osteo.utils.compat import load_model_compat
    model = load_model_compat(model_path)

    batch = _preprocess_patches(patches)
    preds = model.predict(batch, verbose=0)

    indices = list(np.argmax(preds, axis=1))
    majority_idx, _ = Counter(indices).most_common(1)[0]
    avg_probs = preds.mean(axis=0)

    if class_names is None:
        class_names = [str(i) for i in range(len(avg_probs))]

    label = class_names[majority_idx]
    probs = {name: float(avg_probs[i]) for i, name in enumerate(class_names)}
    return {'label': label, 'probs': probs}


def model_predict(model_path: str, patches: list):
    from osteo.utils.compat import load_model_compat
    model = load_model_compat(model_path)
    batch = _preprocess_patches(patches)
    return model.predict(batch, verbose=0)
