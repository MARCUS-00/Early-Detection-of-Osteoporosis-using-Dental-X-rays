"""
MobileNetV2 training — ported verbatim from train_mobilenet.py.

Key hyperparameters (do NOT modify):
  IMG_SIZE      = (128, 128)
  BATCH_SIZE    = 32
  EPOCHS        = 20
  LR            = 0.0001
  Augmentation  = rotation_range=20, width/height_shift=0.2, shear=0.2,
                  zoom=0.2, horizontal_flip=True, fill_mode="nearest"
  Callbacks     = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
                  NOTE: EarlyStopping and ModelCheckpoint are NOT present in the
                  source. class_weight is also NOT used. Do not add them.

Saved model filename: "osteoporosis_mobilenetv2.h5"
Random seeds: numpy, random set for reproducibility; does not alter training.
"""
import random
import numpy as np

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001

_SEED = 42


def train(
    train_dir: str,
    val_dir: str,
    output_path: str = "osteoporosis_mobilenetv2.h5",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    img_size: tuple = IMG_SIZE,
) -> str:
    """
    Train the MobileNetV2 classifier and save as an .h5 file.

    Args:
        train_dir: directory with class sub-folders for training.
        val_dir:   directory with class sub-folders for validation.
        output_path: where to save the trained model.

    Returns:
        Path to the saved model file.

    NOTE: EarlyStopping, ModelCheckpoint, and class_weight are NOT used —
          they were described in documentation but are absent from the source
          training script. This is preserved verbatim.
    """
    # Set seeds — does not alter training logic
    random.seed(_SEED)
    np.random.seed(_SEED)

    import tensorflow as tf  # VERSION NOT VERIFIED — left unpinned

    tf.random.set_seed(_SEED)

    from .model import build_mobilenetv2

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    # Verified folder names: train_dir -> "train/", val_dir -> "valid/" (NOT "val/")
    # Class sub-folders: Normal/, Osteopenia/, Osteoporosis/
    # flow_from_directory assigns indices alphabetically: Normal=0, Osteopenia=1, Osteoporosis=2
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    model = build_mobilenetv2(num_classes=3, img_size=img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        )
        # NOTE: No EarlyStopping, no ModelCheckpoint, no class_weight.
        # These were mentioned in project documentation but are absent from the
        # actual training script (train_mobilenet.py). Preserved verbatim.
    ]

    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    model.save(output_path)
    print(f"Model saved to: {output_path}")
    return output_path
