"""Application configuration, resolved from environment variables at import time.

Environment variables
----------------------
SECRET_KEY      Flask session signing key. Set a strong value in production.
DATA_DIR        Single writable root for uploads, reports, and the SQLite DB.
                Unset -> defaults to ``<repo>/data`` (git-ignored) for local dev.
                Set   -> used verbatim (e.g. ``/app/data`` in Docker / HF Spaces).
DATABASE_URL    Overrides the SQLite URI (e.g. to use managed Postgres).
MODEL_PATH      Overrides the path to the EfficientNetB0 ``.keras`` weights.

Note: storage under DATA_DIR is EPHEMERAL on free-tier hosts (wiped on redeploy).
That is an accepted trade-off for the public demo; the default admin is re-seeded
on every boot, which is why ADMIN_PASSWORD must be set as a secret in production.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent  # .../osteoscan
_ROOT_DIR = _PKG_DIR.parent  # repository root

# --- Security ---------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# --- Writable data locations ------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", _ROOT_DIR / "data"))
INSTANCE_DIR = DATA_DIR / "instance"
UPLOAD_FOLDER = DATA_DIR / "uploads"
REPORT_FOLDER = DATA_DIR / "reports"

SQLALCHEMY_DATABASE_URI = os.environ.get(
    "DATABASE_URL", "sqlite:///" + str(INSTANCE_DIR / "osteo.db")
)

# --- Model ------------------------------------------------------------------
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(_ROOT_DIR / "model" / "osteoporosis_efficientnetb0.keras"))
)

# --- Request / upload limits ------------------------------------------------
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB per upload
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

# --- Domain constants -------------------------------------------------------
# Class order is alphabetical and must match the training label encoding.
CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

# Create writable directories eagerly so the app can serve on first request.
for _directory in (INSTANCE_DIR, UPLOAD_FOLDER, REPORT_FOLDER):
    _directory.mkdir(parents=True, exist_ok=True)
