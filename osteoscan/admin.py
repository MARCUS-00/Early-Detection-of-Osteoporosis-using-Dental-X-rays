"""
admin.py — Admin dashboard blueprint (/admin).
Routes:
  GET  /admin/                      — dashboard: total users, total predictions, recent 10
  GET  /admin/users                 — list all users with role badges
  POST /admin/users/<id>/delete     — delete user (cannot delete self)
  POST /admin/users/<id>/role       — toggle admin/dentist
  GET  /admin/predictions           — all predictions table with patient name, label, date
  POST /admin/predictions/<id>/delete — delete a prediction record
"""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, url_for
from flask_login import current_user

from .auth import admin_required
from .extensions import db
from .models import Prediction, User, get_visible_predictions

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

_USERS_URL = "admin.users"
_PREDS_URL = "admin.predictions"


@admin_bp.route("/", methods=["GET"])
@admin_required
def dashboard():
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    recent = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        recent=recent,
    )


@admin_bp.route("/users", methods=["GET"])
@admin_required
def users():
    all_users = User.query.order_by(User.created_at.asc()).all()
    return render_template("admin/users.html", users=all_users)


@admin_bp.route("/users/<int:uid>/delete", methods=["POST"])
@admin_required
def delete_user(uid: int):
    if uid == current_user.id:
        flash("Cannot delete your own account.", "error")
        return redirect(url_for(_USERS_URL))
    user = db.session.get(User, uid)
    if user:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for(_USERS_URL))


@admin_bp.route("/users/<int:uid>/role", methods=["POST"])
@admin_required
def toggle_role(uid: int):
    user = db.session.get(User, uid)
    if user and uid != current_user.id:
        user.role = "dentist" if user.role == "admin" else "admin"
        db.session.commit()
    return redirect(url_for(_USERS_URL))


@admin_bp.route("/predictions", methods=["GET"])
@admin_required
def predictions():
    all_preds = get_visible_predictions(current_user).all()
    return render_template("admin/predictions.html", predictions=all_preds)


@admin_bp.route("/predictions/<int:pid>/delete", methods=["POST"])
@admin_required
def delete_prediction(pid: int):
    pred = db.session.get(Prediction, pid)
    if pred:
        db.session.delete(pred)
        db.session.commit()
    return redirect(url_for(_PREDS_URL))
