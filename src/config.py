"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TuningConfig:
    enabled: bool
    method: str
    cv_folds: int
    n_iter: int
    param_grid: dict[str, list[Any]]


@dataclass
class TrainConfig:
    data_path: str
    target_column: str
    test_size: float
    random_state: int
    model_type: str
    tuning: TuningConfig
    output_dir: str
    save_predictions_sample_rows: int
    metrics_average: str
    fail_on_validation_errors: bool


REQUIRED_KEYS = {
    "data_path",
    "target_column",
    "test_size",
    "random_state",
    "model_type",
    "tuning",
    "output_dir",
    "save_predictions_sample_rows",
    "metrics_average",
    "fail_on_validation_errors",
}


REQUIRED_TUNING_KEYS = {"enabled", "method", "cv_folds", "n_iter", "param_grid"}


def load_config(config_path: str) -> TrainConfig:
    """Load and validate training YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    tuning_raw = data.get("tuning", {})
    missing_tuning = REQUIRED_TUNING_KEYS - set(tuning_raw.keys())
    if missing_tuning:
        raise ValueError(f"Missing required tuning keys: {sorted(missing_tuning)}")

    tuning_cfg = TuningConfig(
        enabled=bool(tuning_raw["enabled"]),
        method=str(tuning_raw["method"]),
        cv_folds=int(tuning_raw["cv_folds"]),
        n_iter=int(tuning_raw["n_iter"]),
        param_grid=dict(tuning_raw["param_grid"]),
    )

    cfg = TrainConfig(
        data_path=str(data["data_path"]),
        target_column=str(data["target_column"]),
        test_size=float(data["test_size"]),
        random_state=int(data["random_state"]),
        model_type=str(data["model_type"]),
        tuning=tuning_cfg,
        output_dir=str(data["output_dir"]),
        save_predictions_sample_rows=int(data["save_predictions_sample_rows"]),
        metrics_average=str(data["metrics_average"]),
        fail_on_validation_errors=bool(data["fail_on_validation_errors"]),
    )

    if cfg.model_type != "random_forest":
        raise ValueError("Only 'random_forest' model_type is supported in this project.")
    if not (0.0 < cfg.test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")
    if cfg.metrics_average not in {"micro", "macro", "weighted"}:
        raise ValueError("metrics_average must be one of: micro, macro, weighted")
    if cfg.tuning.method not in {"grid", "randomized"}:
        raise ValueError("tuning.method must be 'grid' or 'randomized'")

    return cfg
