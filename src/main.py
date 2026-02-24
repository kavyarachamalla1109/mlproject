"""Main orchestrator for AgriBot MLOps mini pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.artifacts import save_model_artifacts
from src.deploy import create_inference_bundle
from src.config import load_config
from src.evaluate import evaluate_model
from src.preprocess import preprocess_data
from src.train import train_baseline_model
from src.tune import tune_model
from src.utils import ensure_dir, get_environment_info, save_json, setup_logging, utc_timestamp
from src.validate import validate_data_and_config

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str) -> int:
    """Execute full MLOps workflow from config to artifact generation."""
    config = load_config(config_path)
    output_dir = ensure_dir(config.output_dir)

    validation_report = validate_data_and_config(config, str(output_dir))
    if not validation_report.get("validation_passed", False):
        message = f"Validation failed: {validation_report.get('errors', [])}"
        if config.fail_on_validation_errors:
            LOGGER.error(message)
            return 1
        LOGGER.warning(message)

    prepared = preprocess_data(config)
    baseline_model, train_metadata = train_baseline_model(prepared.X_train, prepared.y_train, config)

    final_model, tuning_result = tune_model(baseline_model, prepared.X_train, prepared.y_train, config)

    eval_payload = evaluate_model(
        model=final_model,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        output_dir=str(output_dir),
        average=config.metrics_average,
        sample_rows=config.save_predictions_sample_rows,
    )

    best_params = tuning_result.get("best_params", {})
    run_summary = {
        "timestamp": utc_timestamp(),
        "config_path": config_path,
        "data_path": config.data_path,
        "rows": validation_report.get("rows"),
        "columns": validation_report.get("columns"),
        "validation_passed": validation_report.get("validation_passed"),
        "model_type": config.model_type,
        "tuning": tuning_result,
        "train_metadata": train_metadata,
        "metrics": eval_payload["metrics"],
        "environment": get_environment_info(),
        "artifacts": [
            "agribot_model.pkl",
            "best_params.json",
            "run_summary.json",
            "metrics.json",
            "metrics.md",
            "predictions_sample.csv",
            "data_validation_report.json",
        ],
    }

    artifact_paths = save_model_artifacts(
        model=final_model,
        output_dir=str(output_dir),
        run_summary=run_summary,
        best_params=best_params,
        preprocessor=None,
    )


    bundle_zip_path = create_inference_bundle(str(output_dir))
    run_summary["artifacts"].append(Path(bundle_zip_path).name)
    save_json(run_summary, Path(output_dir) / "run_summary.json")

    LOGGER.info("Pipeline complete. Key metrics: %s", eval_payload["metrics"])
    LOGGER.info("Artifacts saved at %s", Path(config.output_dir).resolve())
    LOGGER.info("Saved files: %s", artifact_paths)
    LOGGER.info("Inference bundle: %s", bundle_zip_path)
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run AgriBot crop recommendation ML pipeline.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to YAML config")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        code = run_pipeline(args.config)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Fatal pipeline error: %s", exc)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
