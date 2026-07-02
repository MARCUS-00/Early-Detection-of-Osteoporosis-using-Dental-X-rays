#!/usr/bin/env python3
"""One-command launcher for OsteoScan.

Usage:
    python run.py

Creates a local virtual environment (.venv), installs the dependencies from
requirements.txt into it, and starts the app at http://127.0.0.1:5000.

Cross-platform (Windows / macOS / Linux). The first run is slow because it
downloads TensorFlow; every run after that reuses the venv and skips the
install (a hash of requirements.txt is used to detect changes), so it starts
almost instantly.

The app ships with safe development defaults for SECRET_KEY and ADMIN_PASSWORD
(see osteoscan/config.py), so no environment setup is needed to run locally.
Default login -> username: admin   password: admin123
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def venv_python() -> Path:
    """Path to the Python executable inside the local virtual environment."""
    if os.name == "nt":  # Windows
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def requirements_marker() -> Path:
    """A sentinel file whose name encodes the current requirements.txt hash.

    Its presence means 'dependencies for this exact requirements.txt are already
    installed', which lets repeat runs skip the (slow) pip install entirely.
    """
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()[:16]
    return VENV_DIR / f".deps-{digest}"


def run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command, and raise on failure."""
    print("[run] $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def ensure_environment() -> Path:
    """Create the venv (if missing) and install deps (if not already current)."""
    py = venv_python()

    if not py.exists():
        print("[run] creating virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        py = venv_python()

    marker = requirements_marker()
    if not marker.exists():
        print("[run] installing dependencies — first run downloads TensorFlow, please wait ...")
        run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        # Clear any stale markers from previous requirements, then record this one.
        for old in VENV_DIR.glob(".deps-*"):
            old.unlink()
        marker.touch()
    else:
        print("[run] dependencies already installed — skipping install.")

    return py


def main() -> None:
    if not REQUIREMENTS.exists():
        sys.exit(f"[run] requirements.txt not found next to run.py (looked in {ROOT}).")

    py = ensure_environment()

    print("\n[run] starting OsteoScan at http://127.0.0.1:5000   (press Ctrl+C to stop)")
    print("[run] default login -> username: admin   password: admin123\n")

    env = dict(os.environ)
    env.setdefault("FLASK_DEBUG", "1")  # auto-reload on code changes during local dev
    subprocess.run([str(py), str(ROOT / "wsgi.py")], check=True, env=env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[run] a command failed with exit code {exc.returncode}. See the error above.")
    except KeyboardInterrupt:
        print("\n[run] stopped.")
