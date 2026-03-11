"""Shared  helpers for pipeline modules."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

