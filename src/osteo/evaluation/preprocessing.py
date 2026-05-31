"""
Phase C — Image preprocessing for the CV evaluation pipeline.

Two confounds identified in the audit are addressed here:
  1. Channel inconsistency: ~25% of Osteoporosis patches are 1-channel; all
     others are 3-channel (but R=G=B grayscale). Loading with color_mode='rgb'
     in flow_from_dataframe makes Keras auto-convert 1ch→3ch. The
     preprocessing_function then explicitly collapses all channels to the mean
     and replicates 3 times, so any residual 3ch non-uniformity is also erased.
  2. Brightness confound: a brightness-only LOSO baseline scores 53.8%
     (vs 33.3% chance). Per-image mean-std standardization (subtract each
     image's own mean, divide by its own std + epsilon) is applied to
     every image in both train and eval. This collapses all inter-class
     brightness differences (verified in confound diagnostic).

IMPORTANT: Do NOT apply rescale=1/255 in ImageDataGenerator when using
this preprocessing_function — standardization replaces the rescale step.
"""

import numpy as np

_EPS = 1e-7


def make_preprocess_fn():
    """
    Return a Keras-compatible preprocessing_function (float32 HxWxC → float32 HxWxC).

    Steps applied to each image (after Keras resize/augment, before model):
      1. Channel collapse: compute channel-mean → (H,W,1); stack 3 times → (H,W,3).
         This ensures identical R=G=B regardless of original storage format.
      2. Per-image standardization: (x - mean) / (std + 1e-7).
         All pixels become mean-zero; brightness differences between images vanish.

    The function intentionally contains no import-time TF dependency; it operates
    on numpy arrays and is compatible with ImageDataGenerator.preprocessing_function.
    """
    def _preprocess(x: np.ndarray) -> np.ndarray:
        # x: float32, shape (H, W, C), values in [0, 255]
        gray = x.mean(axis=2, keepdims=True)          # (H, W, 1)
        x3   = np.concatenate([gray, gray, gray], axis=2)  # (H, W, 3), R=G=B
        mu   = x3.mean()
        sigma = x3.std() + _EPS
        return (x3 - mu) / sigma                      # zero-mean, unit-scale

    return _preprocess


def make_train_datagen(preprocess_fn=None):
    """
    ImageDataGenerator for training folds.

    Augmentation: rotation20/shift0.2/shear0.2/zoom0.2/hflip — unchanged
    from the original source (train_mobilenet.py).
    No rescale (standardization replaces it).
    """
    import tensorflow as tf  # type: ignore

    if preprocess_fn is None:
        preprocess_fn = make_preprocess_fn()

    return tf.keras.preprocessing.image.ImageDataGenerator(
        # no rescale — standardization handles normalisation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=preprocess_fn,
    )


def make_eval_datagen(preprocess_fn=None):
    """
    ImageDataGenerator for evaluation (no augmentation, same preprocessing).
    """
    import tensorflow as tf  # type: ignore

    if preprocess_fn is None:
        preprocess_fn = make_preprocess_fn()

    return tf.keras.preprocessing.image.ImageDataGenerator(
        # no rescale, no augmentation
        preprocessing_function=preprocess_fn,
    )


def make_flow_kwargs(df: "pd.DataFrame", dataset_root: str, datagen, *, shuffle: bool,
                     batch_size: int = 32, img_size: tuple = (128, 128),
                     seed: int = 42) -> dict:
    """
    Return kwargs dict for datagen.flow_from_dataframe().

    Constructs absolute 'filepath' column from relative_path + dataset_root.
    Uses fixed class ordering [Normal, Osteopenia, Osteoporosis] (alphabetical,
    matching the original flow_from_directory ordering).

    Args:
        df:            fold DataFrame (relative_path, class columns required).
        dataset_root:  absolute path to the 100x100 folder.
        datagen:       ImageDataGenerator instance.
        shuffle:       True for train, False for eval.
        batch_size:    default 32.
        img_size:      default (128, 128) matching source model.
        seed:          random seed.

    Returns:
        Dict that can be unpacked into datagen.flow_from_dataframe(**kwargs).
    """
    import pandas as pd
    from pathlib import Path

    work_df = df.copy()
    root = Path(dataset_root)
    work_df["filepath"] = work_df["relative_path"].apply(lambda p: str(root / p))

    return dict(
        dataframe=work_df,
        directory=None,          # absolute paths are in 'filepath' column
        x_col="filepath",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=["Normal", "Osteopenia", "Osteoporosis"],  # fix alphabetical order
        color_mode="rgb",        # Keras auto-converts 1ch→3ch; preprocess_fn normalises
        shuffle=shuffle,
        seed=seed,
        validate_filenames=True,
    )
