"""OsteoScan — osteoporosis screening from dental periapical radiographs.

This package exposes an application factory, :func:`create_app`, which assembles
the Flask app from its parts: configuration, extensions (SQLAlchemy, Flask-Login),
blueprints (``main``, ``auth``, ``admin``), database tables, and a one-time
default-admin seed.

Production servers load the app via ``wsgi.py`` (``gunicorn wsgi:app``).
"""

from __future__ import annotations

import os

from flask import Flask
from flask_login import current_user

from . import config
from .extensions import db, login_manager


def create_app(test_config: dict | None = None) -> Flask:
    """Build and return a fully-configured Flask application.

    Args:
        test_config: Optional dict of config overrides (used by the test suite,
            e.g. to point at a temporary database).

    Returns:
        A ready-to-serve :class:`flask.Flask` instance.
    """
    # __name__ == "osteoscan", so templates/ and static/ resolve inside the package.
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        SQLALCHEMY_DATABASE_URI=config.SQLALCHEMY_DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
    )
    if test_config:
        app.config.update(test_config)

    # Bind extensions to this specific app instance.
    db.init_app(app)
    login_manager.init_app(app)

    # Register blueprints. Imported here (not at module top) to avoid the
    # circular import that would arise from blueprints importing this package.
    from .admin import admin_bp
    from .auth import auth_bp
    from .main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)

    _register_cache_headers(app)

    with app.app_context():
        db.create_all()
        _seed_default_admin()
        _log_startup_paths()

    return app


def _register_cache_headers(app: Flask) -> None:
    """Stop browsers caching authenticated pages (avoids showing stale data
    after logout)."""

    @app.after_request
    def _set_cache_headers(response):
        if current_user.is_authenticated:
            response.headers["Cache-Control"] = "no-store, private"
            response.headers["Pragma"] = "no-cache"
        return response


def _seed_default_admin() -> None:
    """Ensure the default admin account exists and honors ADMIN_PASSWORD.

    The environment variable is used for initial creation and also to reset
    the admin password on every startup in deployed environments. This avoids
    stale credentials when the database is persisted across restarts.
    """
    from .models import User

    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")  # noqa: S105 - dev fallback
    admin = User.query.filter_by(username="admin").first()
    if admin:
        if os.environ.get("ADMIN_PASSWORD"):
            admin.set_password(admin_password)
            db.session.commit()
        return

    admin = User(username="admin", email=None, role="admin")
    admin.set_password(admin_password)
    db.session.add(admin)
    db.session.commit()


def _log_startup_paths() -> None:
    """Print resolved data paths — handy for confirming Docker/HF volume mounts."""
    print(
        f"[startup] DATA_DIR={os.environ.get('DATA_DIR', '(default ./data)')} | "
        f"DB={config.SQLALCHEMY_DATABASE_URI} | "
        f"uploads={config.UPLOAD_FOLDER} | reports={config.REPORT_FOLDER}",
        flush=True,
    )
