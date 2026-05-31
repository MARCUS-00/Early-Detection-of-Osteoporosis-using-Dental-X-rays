"""
Thin config loader — reads YAML files from the configs/ directory.
All hyperparameter defaults are defined in the YAML files, not here.
"""
import yaml
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parents[3] / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config by stem name (e.g. 'classifier', 'yolo', 'unet')."""
    path = _CONFIGS_DIR / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
