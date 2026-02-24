"""Utility helpers for the AgriBot MLOps mini pipeline."""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save dictionary to a JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def markdown_metrics_table(metrics: dict[str, float]) -> str:
    """Convert metrics dict into a Markdown table."""
    header = "| Metric | Value |\n|---|---:|"
    rows = [f"| {k} | {v:.4f} |" for k, v in metrics.items()]
    return "\n".join([header, *rows])


def get_environment_info() -> dict[str, Any]:
    """Collect runtime environment details."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": os.name,
        "cpu_count": os.cpu_count(),
    }
