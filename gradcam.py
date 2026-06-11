import cv2
import numpy as np


def generate_gradcam(
    model_path: str,
    img_path: str,
    output_path: str,
    target_class_idx: int = None,
) -> str:
    import tensorflow as tf
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from osteo.utils.compat import load_model_compat
    model = load_model_compat(model_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read: {img_path}')

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (100, 100))
    img_batch = np.expand_dims(img_resized.astype('float32'), axis=0)

    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer_name = layer.name
            break

    if conv_layer_name is None:
        cv2.imwrite(str(output_path), img_bgr)
        return str(output_path)

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_batch)
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(predictions[0]))
            class_score = predictions[:, target_class_idx]

        grads = tape.gradient(class_score, conv_out)
        if grads is None:
            cv2.imwrite(str(output_path), img_bgr)
            return str(output_path)

        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap, axis=None)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap = heatmap.numpy()

        if heatmap.ndim == 0:
            cv2.imwrite(str(output_path), img_bgr)
            return str(output_path)

        heatmap_resized = cv2.resize(heatmap.astype('float32'), (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)
        cv2.imwrite(str(output_path), overlay)
        return str(output_path)
    except Exception:
        cv2.imwrite(str(output_path), img_bgr)
        return str(output_path)
