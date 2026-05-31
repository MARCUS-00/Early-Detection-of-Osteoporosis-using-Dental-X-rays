"""
Patch-based majority-vote inference — ported verbatim from the inference
section of Complete_implementation.ipynb / train_mobilenet.py sources.

THIS IS THE ACTUAL DEPLOYED INFERENCE PATH.
It does NOT use YOLO, does NOT use U-Net, and does NOT use any ROI detection.
The pipeline is:
  remove_border -> split_into_patches(100) -> ImageDataGenerator(1/255)
  -> model.predict -> argmax -> Counter.most_common -> majority class label

The end-to-end YOLO→U-Net→classifier pipeline is NOT implemented anywhere
in the source code. See pipeline_full.py for a placeholder.
"""
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from osteo.preprocessing.border import remove_border
from osteo.preprocessing.patches import split_into_patches


IMG_SIZE = (128, 128)


def predict_image(
    img_path: str,
    model_path: str,
    class_names: list = None,
    patch_size: int = 100,
) -> str:
    """
    Predict the osteoporosis class for a single dental X-ray image.

    Steps (ported verbatim):
      1. remove_border(img)
      2. split_into_patches(img, patch_size=100)
      3. Write patches to a temporary directory structured for
         flow_from_directory (single class "patches/")
      4. ImageDataGenerator(rescale=1/255).flow_from_directory(
             target_size=(128,128)) -> model.predict
      5. argmax on each patch prediction
      6. Counter.most_common(1) -> majority class index
      7. Return class name.

    NOTE: This uses NEITHER YOLO NOR U-Net. It is the sole verified
    inference path in the codebase.

    Args:
        class_names: list of class labels in the order used during training
                     (e.g. ["Normal", "Osteopenia", "Osteoporosis"]).
                     If None, returns the integer class index as a string.

    Returns:
        Predicted class label (string).
    """
    import tensorflow as tf  # VERSION NOT VERIFIED — left unpinned

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = remove_border(img)
    patches = split_into_patches(img, patch_size=patch_size)

    if not patches:
        raise ValueError(f"No patches extracted from: {img_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_dir = os.path.join(tmpdir, "patches")
        os.makedirs(patch_dir)
        for i, patch in enumerate(patches):
            cv2.imwrite(os.path.join(patch_dir, f"patch_{i:04d}.png"), patch)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        gen = datagen.flow_from_directory(
            tmpdir,
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode=None,
            shuffle=False,
        )

        preds = model_predict(model_path, gen)

    indices = list(np.argmax(preds, axis=1))
    majority_idx, _ = Counter(indices).most_common(1)[0]

    if class_names is not None:
        return class_names[majority_idx]
    return str(majority_idx)


def model_predict(model_path: str, gen):
    """Load model and run predict on generator. Separated for testability."""
    import tensorflow as tf  # VERSION NOT VERIFIED — left unpinned

    model = tf.keras.models.load_model(model_path)
    return model.predict(gen)
