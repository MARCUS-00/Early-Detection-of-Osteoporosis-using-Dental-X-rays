"""
gradcam.py — Grad-CAM heatmap overlay for EfficientNetB0.

Architecture note: EfficientNetB0 is wrapped as a single Functional sub-model
inside the outer model (Input -> RandomFlip -> efficientnetb0 -> Dropout -> Dense x2).
Top-level model.layers therefore contains ZERO Conv2D entries. A shallow layer
search will always return None and silently fall back to copying the original image.

Fix: recursive Conv2D search + sub-model gradient approach so gradients flow
correctly from the outer classification head back through the nested conv layer.
"""

import functools
import logging
import shutil

import cv2
import numpy as np

_log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_model_cached(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_last_conv2d(model):
    """Recursively find the last Conv2D layer in model or any nested sub-model."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if hasattr(layer, "layers"):
            found = _find_last_conv2d(layer)
            if found is not None:
                return found
    return None


def _find_base_submodel(model):
    """Return the first nested Functional/Sequential sub-model (e.g., EfficientNetB0)."""
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 5:
            return layer
    return None


def _split_outer_layers(model, base):
    """Return (pre_layers, post_layers) relative to the nested base sub-model."""
    import tensorflow as tf

    pre, post, seen = [], [], False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is base:
            seen = True
            continue
        (post if seen else pre).append(layer)
    return pre, post


def _compute_gradients_simple(model, last_conv, inp, target_class_idx):
    """Grad-CAM for models where the conv layer sits at the top level."""
    import tensorflow as tf

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        tape.watch(conv_out)
        cls = target_class_idx if target_class_idx is not None else int(tf.argmax(preds[0]))
        score = preds[:, cls]
    return tape.gradient(score, conv_out), conv_out


def _compute_gradients_nested(model, last_conv, inp, target_class_idx):
    """Grad-CAM for models where the conv layer is inside a nested sub-model."""
    import tensorflow as tf

    base = _find_base_submodel(model)
    if base is None:
        raise ValueError("Cannot locate base sub-model containing the Conv2D layer")

    base_grad_model = tf.keras.Model(
        inputs=base.inputs,
        outputs=[last_conv.output, base.output],
        name="base_grad_model",
    )
    pre_layers, post_layers = _split_outer_layers(model, base)

    with tf.GradientTape() as tape:
        x = inp
        for layer in pre_layers:
            x = layer(x, training=False)

        conv_out, base_out = base_grad_model(x)
        tape.watch(conv_out)

        x = base_out
        for layer in post_layers:
            x = layer(x, training=False)
        preds = x

        cls = target_class_idx if target_class_idx is not None else int(tf.argmax(preds[0]))
        score = preds[:, cls]

    return tape.gradient(score, conv_out), conv_out


def _cam_to_heatmap(grads, conv_out, orig_w, orig_h):
    """Pool gradients, build spatial CAM, resize, apply JET colormap."""
    pooled = grads.numpy().mean(axis=(0, 1, 2))  # per-channel weights
    activations = conv_out[0].numpy()  # (H, W, C)
    cam = np.sum(activations * pooled, axis=-1)  # (H, W)

    # Standard Grad-CAM: keep only positive contributions (ReLU)
    cam_pos = np.maximum(cam, 0)
    if cam_pos.max() > 0:
        cam_norm = cam_pos / cam_pos.max()
    else:
        # All locations have negative influence; use magnitude so the overlay
        # still shows WHERE the model focused rather than a uniform blue wash.
        _log.debug("Grad-CAM ReLU output all-zero; using abs-magnitude fallback")
        cam_abs = np.abs(cam)
        cam_norm = cam_abs / cam_abs.max() if cam_abs.max() > 0 else cam_abs

    cam_resized = cv2.resize(cam_norm, (orig_w, orig_h))
    return cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_gradcam(
    model_path: str,
    img_path: str,
    output_path: str,
    target_class_idx: int = None,
    alpha: float = 0.45,
) -> str:
    """
    Generate Grad-CAM overlay and write to output_path.

    Args:
        model_path:       Path to .keras model file.
        img_path:         Input image path.
        output_path:      Where to save the overlay image.
        target_class_idx: Class index for Grad-CAM. Defaults to argmax(pred).
        alpha:            Heatmap blend weight.

    Returns:
        output_path (always; falls back to copying original only when no Conv2D
        layer exists anywhere in the model, logging a warning when it does).
    """
    try:
        import tensorflow as tf

        model = _load_model_cached(str(model_path))

        last_conv = _find_last_conv2d(model)
        if last_conv is None:
            _log.warning(
                "No Conv2D layer found anywhere in the model (checked recursively). "
                "Falling back to copying original image."
            )
            shutil.copy(str(img_path), str(output_path))
            return str(output_path)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = model.input_shape[1:3]
        img_resized = cv2.resize(img_rgb, (w, h)).astype(np.float32)
        inp = tf.constant(np.expand_dims(img_resized, axis=0))

        top_level_conv_names = {
            layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
        }
        if last_conv.name in top_level_conv_names:
            grads, conv_out = _compute_gradients_simple(model, last_conv, inp, target_class_idx)
        else:
            grads, conv_out = _compute_gradients_nested(model, last_conv, inp, target_class_idx)

        if grads is None:
            raise ValueError(
                "GradientTape could not trace a path from the target score "
                "back to the conv layer output (grads is None)."
            )

        heatmap = _cam_to_heatmap(grads, conv_out, orig_w, orig_h)
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
        cv2.imwrite(str(output_path), overlay)

    except Exception:
        _log.exception(
            "Grad-CAM generation failed for %s — copying original image as fallback",
            img_path,
        )
        try:
            shutil.copy(str(img_path), str(output_path))
        except Exception:
            pass

    return str(output_path)
