"""Artifact persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils import save_json


def save_model_artifacts(
    model: Any,
    output_dir: str,
    run_summary: dict[str, Any],
    best_params: dict[str, Any],
    preprocessor: Any | None = None,
) -> dict[str, str]:
    """Save model and metadata artifacts to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "agribot_model.pkl"
    joblib.dump(model, model_path)

    paths: dict[str, str] = {
        "model": str(model_path),
    }

    if preprocessor is not None:
        preprocessor_path = out / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        paths["preprocessor"] = str(preprocessor_path)

    save_json(best_params, out / "best_params.json")
    save_json(run_summary, out / "run_summary.json")

    paths["best_params"] = str(out / "best_params.json")
    paths["run_summary"] = str(out / "run_summary.json")
    return paths
