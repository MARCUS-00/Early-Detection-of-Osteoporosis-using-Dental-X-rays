"""Main blueprint: the user-facing screening workflow.

Routes
------
GET  /                     Upload form (login required).
POST /predict              Validate, run the pipeline, persist, render result.
GET  /history              Current user's predictions (admins see all).
GET  /report               Download a PDF report for a prediction.
GET  /media/uploads/<f>    Serve an uploaded X-ray / Grad-CAM image.
GET  /health               Liveness probe (used by Render/HF health checks).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from . import config
from .extensions import db
from .ml import gradcam, pipeline
from .ml.validate import detect_panoramic_warning, validate_dental_xray
from .models import Prediction, get_visible_predictions
from .reports import generate_report

main_bp = Blueprint("main", __name__)

_INDEX_TEMPLATE = "index.html"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_allowed(filename: str) -> bool:
    """True if the filename has a permitted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def _run_gradcam(uid: str, ext: str, img_path: Path, label: str) -> tuple[str, str | None]:
    """Generate a Grad-CAM overlay for ``label``.

    Returns:
        (gradcam_filename, filename_to_serve_or_None). The second value is None
        when Grad-CAM produced no file (it degrades gracefully on failure).
    """
    class_idx = config.CLASS_NAMES.index(label)
    gradcam_filename = f"gc_{uid}.{ext}"
    gradcam_path = config.UPLOAD_FOLDER / gradcam_filename
    gradcam.generate_gradcam(
        str(config.MODEL_PATH),
        str(img_path),
        str(gradcam_path),
        target_class_idx=class_idx,
    )
    serve_name = gradcam_filename if gradcam_path.exists() else None
    return gradcam_filename, serve_name


def _save_prediction(
    *,
    patient_name: str,
    age: str,
    gender: str,
    label: str,
    probs: dict,
    image_filename: str,
    gradcam_filename: str,
    has_gradcam: bool,
) -> None:
    """Persist a prediction row owned by the current user.

    DB errors are swallowed (rolled back) so a storage hiccup never blocks the
    user from seeing their result.
    """
    try:
        record = Prediction(
            user_id=current_user.id,
            patient_name=patient_name,
            age=age,
            gender=gender,
            label=label,
            prob_normal=probs.get("Normal", 0.0),
            prob_osteopenia=probs.get("Osteopenia", 0.0),
            prob_osteoporosis=probs.get("Osteoporosis", 0.0),
            image_filename=image_filename,
            gradcam_filename=gradcam_filename if has_gradcam else "",
        )
        db.session.add(record)
        db.session.commit()
    except Exception:
        db.session.rollback()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@main_bp.route("/", methods=["GET"])
@login_required
def index():
    """Render the upload form."""
    return render_template(_INDEX_TEMPLATE)


@main_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    """Validate the upload, run the inference pipeline, and render the result."""
    if "xray" not in request.files:
        return render_template(_INDEX_TEMPLATE, error="No file uploaded.")

    file = request.files["xray"]
    if not file or not file.filename or not _is_allowed(file.filename):
        return render_template(_INDEX_TEMPLATE, error="Invalid or missing file.")

    patient_name = request.form.get("patient_name", "Unknown").strip() or "Unknown"
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()

    # Persist the upload under a unique, collision-free name.
    uid = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    image_filename = f"{uid}.{ext}"
    img_path = config.UPLOAD_FOLDER / image_filename
    file.save(str(img_path))

    # Reject anything that clearly is not a dental X-ray before wasting compute.
    is_valid, reason = validate_dental_xray(str(img_path))
    if not is_valid:
        img_path.unlink(missing_ok=True)
        return render_template(_INDEX_TEMPLATE, error=reason)

    panoramic_warning = detect_panoramic_warning(str(img_path))

    if not config.MODEL_PATH.exists():
        img_path.unlink(missing_ok=True)
        return render_template(
            _INDEX_TEMPLATE,
            error="Model file not found. Add osteoporosis_efficientnetb0.keras to model/.",
        )

    try:
        result = pipeline.run_pipeline(str(img_path), str(config.MODEL_PATH))
    except Exception as exc:
        return render_template(_INDEX_TEMPLATE, error=f"Inference failed: {exc}")

    label = result["label"]
    probs = result["probs"]
    gradcam_filename, gradcam_serve = _run_gradcam(uid, ext, img_path, label)

    _save_prediction(
        patient_name=patient_name,
        age=age,
        gender=gender,
        label=label,
        probs=probs,
        image_filename=image_filename,
        gradcam_filename=gradcam_filename,
        has_gradcam=gradcam_serve is not None,
    )

    return render_template(
        "result.html",
        prediction=label,
        probs=probs,
        image_rel=image_filename,
        gradcam_rel=gradcam_serve,
        patient_name=patient_name,
        img_filename=image_filename,
        gradcam_filename=gradcam_filename if gradcam_serve else "",
        age=age,
        gender=gender,
        patch_count=result.get("patch_count", 0),
        stages=result.get("stages", {}),
        roi_count=result.get("roi_count", 1),
        panoramic_warning=panoramic_warning,
    )


@main_bp.route("/history", methods=["GET"])
@login_required
def history():
    """Show predictions visible to the current user (all rows for admins)."""
    predictions = get_visible_predictions(current_user).all()
    return render_template(
        "history.html",
        predictions=predictions,
        is_admin=(current_user.role == "admin"),
    )


@main_bp.route("/report", methods=["GET"])
@login_required
def download_report():
    """Generate and download a one-page PDF report for a prediction."""
    patient_name = request.args.get("patient_name", "Unknown")
    label = request.args.get("label", "Unknown")
    image_filename = request.args.get("img_filename", "")
    gradcam_filename = request.args.get("gradcam_filename", "")
    age = request.args.get("age", "")
    gender = request.args.get("gender", "")

    image_filename = secure_filename(image_filename) if image_filename else ""
    gradcam_filename = secure_filename(gradcam_filename) if gradcam_filename else ""

    probs = {
        "Normal": float(request.args.get("Normal", 0)),
        "Osteopenia": float(request.args.get("Osteopenia", 0)),
        "Osteoporosis": float(request.args.get("Osteoporosis", 0)),
    }

    image_path = str(config.UPLOAD_FOLDER / image_filename) if image_filename else None
    gradcam_path = str(config.UPLOAD_FOLDER / gradcam_filename) if gradcam_filename else None

    report_path = str(config.REPORT_FOLDER / f"report_{uuid.uuid4().hex}.pdf")
    try:
        generate_report(
            report_path,
            patient_name,
            label,
            probs,
            image_path=image_path,
            gradcam_path=gradcam_path,
            age=age,
            gender=gender,
        )
    except Exception as exc:
        return jsonify({"error": f"Report generation failed: {exc}"}), 500

    safe_name = patient_name.replace(" ", "_")
    return send_file(report_path, as_attachment=True, download_name=f"osteo_report_{safe_name}.pdf")


@main_bp.route("/media/uploads/<path:filename>", methods=["GET"])
@login_required
def serve_upload(filename):
    """Serve an uploaded X-ray or Grad-CAM image from the data directory."""
    return send_from_directory(config.UPLOAD_FOLDER, filename)


@main_bp.route("/health", methods=["GET"])
def health():
    """Liveness probe used by the platform health check."""
    return jsonify({"status": "ok", "model_loaded": config.MODEL_PATH.exists()})


@main_bp.app_errorhandler(413)
def _too_large(_error):
    """Friendly message when an upload exceeds MAX_CONTENT_LENGTH."""
    return (
        render_template(_INDEX_TEMPLATE, error="File too large. Maximum upload size is 16 MB."),
        413,
    )
