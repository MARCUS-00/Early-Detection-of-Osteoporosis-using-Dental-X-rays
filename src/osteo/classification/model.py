"""
MobileNetV2 classifier — ported verbatim from train_mobilenet.py.

Architecture:
  MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dense(128, "relu")
  -> Dropout(0.5)
  -> Dense(3, "softmax")

Fine-tuning: base.trainable=False initially; last 20 layers set trainable=True
before compile.
"""


def build_mobilenetv2(num_classes: int = 3, img_size: tuple = (128, 128)):
    """
    Build and return the MobileNetV2-based classifier.

    The base is loaded with ImageNet weights; the top 20 layers are unfrozen
    for fine-tuning. All other layers remain frozen.

    Source: train_mobilenet.py — ported verbatim.
    """
    import tensorflow as tf  # type: ignore  # VERSION NOT VERIFIED — left unpinned

    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base.trainable = False
    for layer in base.layers[-20:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model
