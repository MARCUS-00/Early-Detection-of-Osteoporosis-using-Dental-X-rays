"""WSGI entry point.

Production servers import ``app`` from here, e.g. ``gunicorn wsgi:app``.
Running this file directly starts Flask's development server (local use only).
"""

from __future__ import annotations

import os

from osteoscan import create_app

app = create_app()

if __name__ == "__main__":
    # Development server only — production serving is handled by gunicorn.
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
