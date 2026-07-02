"""
reports.py - PDF diagnostic report generator (fpdf2).

Produces a single-page A4 PDF:
  - Header: title + institution + rule
  - Patient info table (name, age, gender, date, analysis method)
  - Diagnosis badge (colour-coded by prediction)
  - Confidence score bars (3 rows)
  - Side-by-side images: original X-ray | Grad-CAM (if provided)
  - Disclaimer footer
"""

from __future__ import annotations

import os
from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos

CLASS_COLORS = {
    "Normal": (34, 139, 34),
    "Osteopenia": (200, 110, 0),
    "Osteoporosis": (200, 20, 50),
}

DISCLAIMER = (
    "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE. "
    "N=13 source X-rays; accuracy 61.5% (8/13 LOSO). "
    "Do not use for diagnosis or treatment decisions."
)


def _header(pdf: FPDF) -> None:
    pdf.set_fill_color(26, 39, 68)
    pdf.rect(0, 0, 210, 18, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(10, 4)
    pdf.cell(0, 10, "Osteoporosis Screening Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(10, 11)
    pdf.cell(0, 5, "PES University - Early Detection of Osteoporosis using Dental X-rays")
    pdf.set_text_color(40, 40, 40)
    pdf.set_draw_color(14, 165, 160)
    pdf.set_line_width(0.8)
    pdf.line(10, 20, 200, 20)


def _patient_table(pdf: FPDF, patient_name: str, age: str, gender: str, report_date: str) -> None:
    pdf.set_xy(10, 26)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(26, 39, 68)
    pdf.cell(0, 7, "Patient Information", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    rows = [
        ("Patient Name", patient_name or "Unknown"),
        ("Age", age or "-"),
        ("Gender", gender or "-"),
        ("Report Date", report_date),
        ("Institution", "PES University, Bengaluru - Dept. of Computer Science"),
        ("Methodology", "EfficientNetB0 patch majority-vote (100x100 px patches, CLAHE)"),
        ("Dataset", "Mendeley dental periapical radiographs, 13 sources, LOSO CV"),
    ]
    for label, value in rows:
        pdf.set_x(10)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(45, 6, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _diagnosis_badge(pdf: FPDF, prediction: str) -> None:
    r, g, b = CLASS_COLORS.get(prediction, (100, 100, 100))
    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 12, f"  Diagnosis: {prediction}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)


def _confidence_bars(pdf: FPDF, confidence: dict) -> None:
    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(26, 39, 68)
    pdf.cell(0, 6, "Confidence Scores:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    bar_max = 95
    for cls in ["Normal", "Osteopenia", "Osteoporosis"]:
        prob = confidence.get(cls, 0.0)
        filled = bar_max * prob
        pdf.set_x(10)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(38, 5, cls, new_x=XPos.RIGHT, new_y=YPos.TOP)
        x, y2 = pdf.get_x(), pdf.get_y() + 1
        pdf.set_fill_color(220, 220, 220)
        pdf.rect(x, y2, bar_max, 3, "F")
        cr, cg, cb = CLASS_COLORS.get(cls, (100, 100, 100))
        pdf.set_fill_color(cr, cg, cb)
        if filled > 0:
            pdf.rect(x, y2, filled, 3, "F")
        pdf.set_xy(x + bar_max + 2, pdf.get_y())
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(20, 5, f"{prob * 100:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)


def _images(pdf: FPDF, image_path: str | None, gradcam_path: str | None) -> None:
    if not image_path and not gradcam_path:
        return
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(26, 39, 68)
    pdf.cell(0, 6, "X-ray Analysis:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    y_img = pdf.get_y()
    if image_path and os.path.isfile(image_path):
        try:
            pdf.image(str(image_path), x=10, y=y_img, w=88)
        except Exception:
            pass
    if gradcam_path and os.path.isfile(gradcam_path):
        try:
            pdf.image(str(gradcam_path), x=112, y=y_img, w=88)
        except Exception:
            pass


def _footer(pdf: FPDF) -> None:
    pdf.set_xy(10, 272)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(10, 272, 200, 272)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.set_xy(10, 274)
    pdf.multi_cell(0, 4, DISCLAIMER)


def generate_report(
    output_path: str,
    patient_name: str,
    prediction: str,
    confidence: dict,
    image_path: str = None,
    gradcam_path: str = None,
    age: str = "",
    gender: str = "",
    report_date: str = None,
) -> str:
    """Generate PDF report and write to output_path. Returns output_path."""
    if report_date is None:
        report_date = date.today().isoformat()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    _header(pdf)
    _patient_table(pdf, patient_name, age, gender, report_date)
    _diagnosis_badge(pdf, prediction)
    _confidence_bars(pdf, confidence)
    _images(pdf, image_path, gradcam_path)
    _footer(pdf)

    pdf.output(str(output_path))
    return str(output_path)
