"""Hyperparameter tuning module."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import TrainConfig

LOGGER = logging.getLogger(__name__)


def tune_model(
    baseline_model: RandomForestClassifier,
    X_train: Any,
    y_train: Any,
    config: TrainConfig,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Tune model with GridSearchCV or RandomizedSearchCV when enabled."""
    if not config.tuning.enabled:
        return baseline_model, {
            "tuning_enabled": False,
            "best_params": baseline_model.get_params(),
            "best_cv_score": None,
            "method": None,
            "cv_folds_used": None,
        }

    y_series = pd.Series(y_train)
    min_class_count = int(y_series.value_counts().min()) if not y_series.empty else 1
    cv_folds = max(2, min(config.tuning.cv_folds, min_class_count))

    if cv_folds != config.tuning.cv_folds:
        LOGGER.warning(
            "Adjusted CV folds from %s to %s due to small class counts.",
            config.tuning.cv_folds,
            cv_folds,
        )

    estimator = RandomForestClassifier(random_state=config.random_state)
    if config.tuning.method == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=config.tuning.param_grid,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=config.tuning.param_grid,
            n_iter=config.tuning.n_iter,
            cv=cv_folds,
            scoring="f1_weighted",
            random_state=config.random_state,
            n_jobs=-1,
        )

    search.fit(X_train, y_train)
    best_model: RandomForestClassifier = search.best_estimator_

    tuning_result = {
        "tuning_enabled": True,
        "method": config.tuning.method,
        "cv_folds_used": cv_folds,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
    }
    return best_model, tuning_result
