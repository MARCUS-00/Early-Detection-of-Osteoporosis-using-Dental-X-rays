'''
YOLO -> Keras classifier end-to-end pipeline.

Architecture confirmed by inspecting osteoporosis_mobilenetv2.h5:
  Input: (None, 100, 100, 3) raw uint8 values 0-255
  Internal: Rescaling layer (to_0_1) + U-Net attention (TF) + EfficientNetB0
  Output: (None, 3)  [Normal, Osteopenia, Osteoporosis]

Pipeline:
  1. YOLO detects left and right mandibular cortex ROI boxes
  2. Both ROIs are cropped from the panoramic X-ray
  3. Each crop is split into 100x100 patches (same as predict_patches.py)
  4. All patches are fed to the Keras classifier
  5. Majority vote gives the final label

Falls back to patch-based inference on full image if YOLO weights not provided.
'''
import cv2
import numpy as np
from collections import Counter


def run_full_pipeline(
    img_path: str,
    classifier_path: str,
    class_names: list = None,
    yolo_weights: str = None,
    conf: float = 0.1,
) -> dict:
    '''
    Run the complete pipeline on a single dental panoramic X-ray.

    Args:
        img_path:         Path to input X-ray image.
        classifier_path:  Path to osteoporosis_mobilenetv2.h5
        class_names:      List of class names. Defaults to
                          ['Normal', 'Osteopenia', 'Osteoporosis']
        yolo_weights:     Path to trained YOLO .pt weights.
                          If None or not found, falls back to full-image
                          patch inference (same as predict_patches.py).
        conf:             YOLO confidence threshold.

    Returns:
        dict with keys:
          label (str): predicted class name
          probs (dict): {class_name: float} averaged across patches
          mode  (str): 'yolo' or 'fallback'
    '''
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))

    if class_names is None:
        class_names = ['Normal', 'Osteopenia', 'Osteoporosis']

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    # Decide whether to use YOLO or fall back
    use_yolo = (
        yolo_weights is not None
        and os.path.exists(str(yolo_weights))
    )

    if use_yolo:
        try:
            from osteo.roi.yolo_extract import roi_extraction
            left_box, right_box = roi_extraction(img_path, yolo_weights, conf=conf)
            crops = []
            for box in (left_box, right_box):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    crops.append(crop)
            mode = 'yolo'
        except Exception:
            # If YOLO fails (no detections, etc.) fall back gracefully
            crops = [img]
            mode = 'fallback'
    else:
        crops = [img]
        mode = 'fallback'

    from osteo.preprocessing.border import remove_border
    from osteo.preprocessing.patches import split_into_patches
    from osteo.utils.compat import load_model_compat

    all_patches = []
    for crop in crops:
        crop = remove_border(crop)
        patches = split_into_patches(crop, patch_size=100)
        all_patches.extend(patches)

    if not all_patches:
        # Last resort: resize the whole image to 100x100 as a single patch
        fallback = remove_border(img)
        all_patches = [cv2.resize(fallback, (100, 100))]

    # Preprocess: BGR->RGB, raw float32 (model rescales internally)
    batch = []
    for p in all_patches:
        p = cv2.resize(p, (100, 100))
        if len(p.shape) == 2:
            p = np.stack([p, p, p], axis=-1)
        elif p.shape[2] == 1:
            p = np.concatenate([p, p, p], axis=-1)
        else:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        batch.append(p.astype(np.float32))
    batch = np.array(batch)

    model = load_model_compat(classifier_path)
    preds = model.predict(batch, verbose=0)

    indices = list(np.argmax(preds, axis=1))
    majority_idx, _ = Counter(indices).most_common(1)[0]
    avg_probs = preds.mean(axis=0)

    label = class_names[majority_idx]
    probs = {name: float(avg_probs[i]) for i, name in enumerate(class_names)}

    return {'label': label, 'probs': probs, 'mode': mode}
