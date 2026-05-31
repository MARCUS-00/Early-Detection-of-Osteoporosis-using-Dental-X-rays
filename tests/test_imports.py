"""
Smoke test: verify all osteo modules import without error.
No dataset files or model weights are needed.
"""
import importlib
import pytest


MODULES = [
    "osteo",
    "osteo.config",
    "osteo.utils.io",
    "osteo.preprocessing",
    "osteo.preprocessing.border",
    "osteo.preprocessing.patches",
    "osteo.roi",
    "osteo.roi.crop_from_labels",
    "osteo.roi.yolo_extract",
    "osteo.roi.yolo_train",
    "osteo.segmentation",
    "osteo.segmentation.masks_classical",
    "osteo.segmentation.dataset",
    "osteo.segmentation.unet",
    "osteo.segmentation.train_unet",
    "osteo.segmentation.experimental_vit",
    "osteo.classification",
    "osteo.classification.model",
    "osteo.classification.train",
    "osteo.classification.evaluate",
    "osteo.inference",
    "osteo.inference.predict_patches",
    "osteo.inference.pipeline_full",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
