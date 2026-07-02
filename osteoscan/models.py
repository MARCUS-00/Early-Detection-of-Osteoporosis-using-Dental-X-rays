"""Database models.

User  — an account (``admin`` or ``dentist``) with a hashed password.
Prediction — one screening result, owned by a User (row-level access control).
"""

from __future__ import annotations

from datetime import UTC, datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=False, default="")
    role = db.Column(db.String(16), nullable=False, default="dentist")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    predictions = db.relationship(
        "Prediction", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_name = db.Column(db.String(128), default="Unknown")
    age = db.Column(db.String(16), default="")
    gender = db.Column(db.String(32), default="")
    label = db.Column(db.String(32), nullable=False)
    prob_normal = db.Column(db.Float, default=0.0)
    prob_osteopenia = db.Column(db.Float, default=0.0)
    prob_osteoporosis = db.Column(db.Float, default=0.0)
    image_filename = db.Column(db.String(256), default="")
    gradcam_filename = db.Column(db.String(256), default="")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))


def get_visible_predictions(user):
    """Return Prediction query scoped by role: admins see all, dentists see only their own."""
    q = Prediction.query.order_by(Prediction.created_at.desc())
    if user.role != "admin":
        q = q.filter_by(user_id=user.id)
    return q
