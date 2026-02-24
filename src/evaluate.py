"""Model evaluation and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.utils import markdown_metrics_table, save_json


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
    average: str,
    sample_rows: int,
) -> dict[str, Any]:
    """Evaluate model performance and save evaluation artifacts."""
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, preds, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, preds, average=average, zero_division=0)),
    }

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    payload = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    out = Path(output_dir)
    save_json(payload, out / "metrics.json")

    md_content = "# Evaluation Metrics\n\n"
    md_content += markdown_metrics_table(metrics)
    md_content += "\n\n## Confusion Matrix\n\n"
    md_content += str(cm.tolist())
    (out / "metrics.md").write_text(md_content, encoding="utf-8")

    sample_df = X_test.copy()
    sample_df["actual"] = y_test.values
    sample_df["predicted"] = preds
    sample_df.head(sample_rows).to_csv(out / "predictions_sample.csv", index=False)

    return payload
