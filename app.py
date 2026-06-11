import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from pathlib import Path
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'osteoporosis_mobilenetv2.h5')
UPLOAD_DIR   = Path('static/uploads')
REPORTS_DIR  = Path('static/reports')
CLASS_NAMES  = ['Normal', 'Osteopenia', 'Osteoporosis']
ALLOWED_EXTS = {'jpg', 'jpeg', 'png', 'bmp'}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['xray']
    if not file or file.filename == '' or not allowed(file.filename):
        return jsonify({'error': 'Invalid or missing file'}), 400

    patient_name = request.form.get('patient_name', 'Unknown').strip() or 'Unknown'

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = uuid.uuid4().hex
    img_filename = f'{uid}.{ext}'
    img_path = str(UPLOAD_DIR / img_filename)
    file.save(img_path)

    try:
        import os
        yolo_weights = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'yolo_weights.pt'
        )
        from osteo.inference.pipeline_full import run_full_pipeline
        result = run_full_pipeline(
            img_path=img_path,
            classifier_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            yolo_weights=yolo_weights if os.path.exists(yolo_weights) else None,
        )
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    label = result['label']
    probs = result['probs']
    class_idx = CLASS_NAMES.index(label)

    gradcam_filename = None
    gradcam_rel = None
    try:
        from gradcam import generate_gradcam
        gradcam_filename = f'gc_{uid}.{ext}'
        gradcam_abs = str(UPLOAD_DIR / gradcam_filename)
        generate_gradcam(MODEL_PATH, img_path, gradcam_abs, target_class_idx=class_idx)
        gradcam_rel = f'uploads/{gradcam_filename}'
    except Exception:
        gradcam_filename = None
        gradcam_rel = None

    return render_template(
        'result.html',
        prediction=label,
        probs=probs,
        image_rel=f'uploads/{img_filename}',
        gradcam_rel=gradcam_rel,
        patient_name=patient_name,
        img_filename=img_filename,
        gradcam_filename=gradcam_filename or '',
    )


@app.route('/report')
def download_report():
    patient_name     = request.args.get('patient_name', 'Unknown')
    label            = request.args.get('label', 'Unknown')
    img_filename     = request.args.get('img_filename', '')
    gradcam_filename = request.args.get('gradcam_filename', '')

    probs = {
        'Normal':       float(request.args.get('Normal',       0)),
        'Osteopenia':   float(request.args.get('Osteopenia',   0)),
        'Osteoporosis': float(request.args.get('Osteoporosis', 0)),
    }

    img_path     = str(UPLOAD_DIR / img_filename)     if img_filename     else None
    gradcam_path = str(UPLOAD_DIR / gradcam_filename) if gradcam_filename else None

    report_name = f'report_{uuid.uuid4().hex}.pdf'
    report_path = str(REPORTS_DIR / report_name)

    try:
        from report_generator import generate_report
        generate_report(report_path, patient_name, label, probs, img_path, gradcam_path)
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {e}'}), 500

    safe_name = patient_name.replace(' ', '_')
    return send_file(report_path, as_attachment=True,
                     download_name=f'osteo_report_{safe_name}.pdf')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
