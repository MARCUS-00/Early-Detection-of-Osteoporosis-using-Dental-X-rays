"""Test suite for OsteoScan.

Covers HTTP endpoints (via a test client on a temporary database) and the
image-processing / validation helpers. The single model-dependent test is
skipped automatically when the weights file is absent.
"""

import json

import numpy as np
import pytest

from osteoscan import create_app
from osteoscan.config import MODEL_PATH
from osteoscan.ml.inference import (
    apply_clahe,
    preprocess_patches,
    remove_border,
    split_into_patches,
)
from osteoscan.ml.validate import validate_dental_xray
from osteoscan.reports import generate_report


def _tf_available() -> bool:
    """True if TensorFlow can be imported (it is an optional/heavy dependency)."""
    import importlib.util

    return importlib.util.find_spec("tensorflow") is not None


def _model_available() -> bool:
    """True only when the model weights file exists and is large enough.

    A git-LFS pointer stub checked out without `lfs: true` is only ~130 bytes.
    The real EfficientNetB0 weights are ~18 MB, so this prevents CI from
    attempting inference on a stub file.
    """
    return MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000


@pytest.fixture
def client(tmp_path):
    """A test client backed by an isolated, temporary SQLite database."""
    app = create_app(
        {
            "TESTING": True,
            "LOGIN_DISABLED": True,  # bypass @login_required for endpoint tests
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        }
    )
    with app.test_client() as test_client:
        yield test_client


def _blank_image(path, size=(200, 200), value=128):
    """Write a flat, single-colour test image to ``path``."""
    import cv2

    cv2.imwrite(str(path), np.full((*size, 3), value, dtype=np.uint8))


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"


def test_index_loads(client):
    assert client.get("/").status_code == 200


def test_predict_no_file(client):
    resp = client.post("/predict", data={})
    assert resp.status_code == 200
    assert b"No file uploaded" in resp.data


def test_predict_blank_image(client, tmp_path):
    img = tmp_path / "blank.png"
    _blank_image(img)
    with open(img, "rb") as fh:
        resp = client.post("/predict", data={"xray": (fh, "blank.png")})
    assert resp.status_code == 200  # rejected by the validator, not crashed


def test_predict_colour_image(client, tmp_path):
    import cv2

    img = tmp_path / "colour.png"
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # strong blue channel -> looks like a colour photo
    cv2.imwrite(str(img), arr)
    with open(img, "rb") as fh:
        resp = client.post("/predict", data={"xray": (fh, "colour.png")})
    assert resp.status_code == 200


def test_remove_border():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:80, 20:80] = 200
    cropped = remove_border(arr)
    assert cropped.shape[0] <= 100 and cropped.shape[1] <= 100


def test_split_into_patches():
    arr = np.full((400, 800, 3), 128, dtype=np.uint8)
    patches = split_into_patches(arr, patch_size=100)
    assert len(patches) == 32  # (400 // 100) * (800 // 100)


def test_apply_clahe_shape():
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert apply_clahe(arr).shape == (100, 100, 3)


def test_preprocess_patches_shape():
    patches = [np.full((100, 100, 3), 128, dtype=np.uint8) for _ in range(3)]
    batch = preprocess_patches(patches)
    assert batch.shape == (3, 100, 100, 3)


def test_validate_rejects_blank(tmp_path):
    img = tmp_path / "blank.png"
    _blank_image(img)
    is_valid, _ = validate_dental_xray(str(img))
    assert is_valid is False


def test_validate_rejects_small(tmp_path):
    img = tmp_path / "small.png"
    _blank_image(img, size=(40, 40))
    is_valid, _ = validate_dental_xray(str(img))
    assert is_valid is False


def test_validate_accepts_xray_like(tmp_path):
    """A grayscale image with realistic tonal variation should pass."""
    import cv2

    rng = np.random.default_rng(0)
    arr = rng.integers(30, 220, size=(200, 200), dtype=np.uint8)
    img = tmp_path / "xray.png"
    cv2.imwrite(str(img), arr)
    is_valid, _ = validate_dental_xray(str(img))
    assert is_valid is True


def test_report_generation(tmp_path):
    out = tmp_path / "report.pdf"
    generate_report(
        str(out),
        "Test Patient",
        "Osteopenia",
        {"Normal": 0.2, "Osteopenia": 0.6, "Osteoporosis": 0.2},
    )
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(
    not _model_available() or not _tf_available(),
    reason="model weights (real, not LFS pointer) or TensorFlow not available",
)
def test_inference_pipeline(tmp_path):
    import cv2

    from osteoscan.ml import pipeline

    rng = np.random.default_rng(1)
    arr = rng.integers(30, 220, size=(200, 200), dtype=np.uint8)
    img = tmp_path / "xray.png"
    cv2.imwrite(str(img), arr)
    result = pipeline.run_pipeline(str(img), str(MODEL_PATH))
    assert result["label"] in ("Normal", "Osteopenia", "Osteoporosis")
