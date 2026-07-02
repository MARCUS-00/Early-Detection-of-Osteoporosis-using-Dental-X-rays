"""Flask extension singletons.

These are instantiated here *without* an application so that:
  * the application factory (`create_app`) can bind them to a specific app, and
  * other modules can import ``db`` / ``login_manager`` without importing the
    factory (which would create a circular import).
"""

from __future__ import annotations

from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

# SQLAlchemy ORM handle — bound to an app in create_app() via db.init_app(app).
db = SQLAlchemy()

# Flask-Login session manager — bound to an app in create_app().
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"
