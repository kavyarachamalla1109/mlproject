"""Baseline model training module."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.config import TrainConfig


def train_baseline_model(X_train: Any, y_train: Any, config: TrainConfig) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train baseline RandomForestClassifier."""
    model = RandomForestClassifier(random_state=config.random_state)
    model.fit(X_train, y_train)

    metadata = {
        "model_type": config.model_type,
        "n_features": int(X_train.shape[1]),
        "n_train_rows": int(X_train.shape[0]),
        "classes": sorted(list(set(y_train))),
    }
    return model, metadata
