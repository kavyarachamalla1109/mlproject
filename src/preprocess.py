"""Data preprocessing and split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TrainConfig


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    preprocessor_name: str


def preprocess_data(config: TrainConfig) -> PreparedData:
    """Load dataset, split features and target, and create train/test sets."""
    df = pd.read_csv(config.data_path)

    feature_columns = [c for c in df.columns if c != config.target_column]
    X = df[feature_columns].copy()
    y = df[config.target_column].copy()

    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        preprocessor_name="identity",
    )
