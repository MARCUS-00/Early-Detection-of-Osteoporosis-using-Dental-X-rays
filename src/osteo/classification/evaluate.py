"""
MobileNetV2 evaluation — ported verbatim from test_mobilenet.py.

# DISCREPANCY: The original test_mobilenet.py loaded the model from
# "osteoporosis_mobilenet_model.h5" while train_mobilenet.py saved to
# "osteoporosis_mobilenetv2.h5". This file standardises on the trained
# filename ("osteoporosis_mobilenetv2.h5"). The original name is noted here
# for reference only.

NOTE: The accuracy computed here is PER-PATCH accuracy, not per-patient
or per-image. Patches are extracted from dental X-rays; a single image
produces many patches. This metric does not correspond to clinical accuracy.
"""

IMG_SIZE = (128, 128)
BATCH_SIZE = 64  # evaluation batch size differs from training (32 vs 64)


def evaluate(
    test_dir: str,
    model_path: str = "osteoporosis_mobilenetv2.h5",
    # DISCREPANCY: original test_mobilenet.py used "osteoporosis_mobilenet_model.h5"
    # Standardised to the name produced by train_mobilenet.py.
    batch_size: int = BATCH_SIZE,
    img_size: tuple = IMG_SIZE,
) -> dict:
    """
    Evaluate the saved MobileNetV2 model on a test directory.

    Returns a dict with keys: "accuracy", "classification_report".

    NOTE: accuracy is PER-PATCH — see module docstring.
    Source: test_mobilenet.py — ported verbatim (filename standardised).
    """
    import tensorflow as tf  # VERSION NOT VERIFIED — left unpinned
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score

    model = tf.keras.models.load_model(model_path)

    # Verified folder name: test_dir -> "test/"
    # Class sub-folders: Normal/, Osteopenia/, Osteoporosis/
    # flow_from_directory assigns indices alphabetically: Normal=0, Osteopenia=1, Osteoporosis=2
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,  # must be False for y_true alignment
    )

    y_true = test_gen.classes
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)

    # NOTE: accuracy below is PER-PATCH, not per-patient/image.
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(test_gen.class_indices.keys()),
    )

    print(f"Per-patch accuracy: {acc:.4f}")
    print(report)

    return {"accuracy": acc, "classification_report": report}
