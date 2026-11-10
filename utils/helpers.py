"""Helper utilities for file paths, configuration loading, and project root resolution in SocialPulse."""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and parse the project YAML configuration file."""
    root = get_project_root()
    full_path = root / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

