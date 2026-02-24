"""Dataset and config validation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TrainConfig
from src.utils import save_json, utc_timestamp


REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]


def validate_data_and_config(config: TrainConfig, output_dir: str) -> dict[str, Any]:
    """Validate dataset integrity and produce a JSON report."""
    report: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "data_path": config.data_path,
        "target_column": config.target_column,
        "errors": [],
        "warnings": [],
        "validation_passed": False,
    }

    path = Path(config.data_path)
    if not path.exists():
        report["errors"].append(f"Data file does not exist: {config.data_path}")
        _save_report(report, output_dir)
        return report

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        report["errors"].append(f"Unable to read CSV: {exc}")
        _save_report(report, output_dir)
        return report

    report["rows"] = int(df.shape[0])
    report["columns"] = int(df.shape[1])

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        report["errors"].append(f"Missing required columns: {missing_columns}")

    if config.target_column not in df.columns:
        report["errors"].append(f"Target column not found: {config.target_column}")

    report["null_counts"] = {col: int(count) for col, count in df.isnull().sum().items()}
    report["duplicate_rows"] = int(df.duplicated().sum())

    numeric_features = [c for c in REQUIRED_COLUMNS if c != "label"]
    coercion_failures: dict[str, int] = {}
    for col in numeric_features:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            failures = int(coerced.isnull().sum() - df[col].isnull().sum())
            if failures > 0:
                coercion_failures[col] = failures

    report["numeric_coercion_failures"] = coercion_failures
    if coercion_failures:
        report["errors"].append(f"Numeric coercion failures detected: {coercion_failures}")

    if config.target_column in df.columns:
        class_dist = df[config.target_column].value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}

    report["validation_passed"] = len(report["errors"]) == 0
    _save_report(report, output_dir)
    return report


def _save_report(report: dict[str, Any], output_dir: str) -> None:
    save_json(report, Path(output_dir) / "data_validation_report.json")
