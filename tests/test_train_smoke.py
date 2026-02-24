from pathlib import Path

from src.main import run_pipeline


def test_train_pipeline_smoke(tmp_path: Path) -> None:
    config_text = """
data_path: data/raw/crop_recommendation_sample.csv
target_column: label
test_size: 0.25
random_state: 7
model_type: random_forest
tuning:
  enabled: false
  method: grid
  cv_folds: 3
  n_iter: 4
  param_grid:
    n_estimators: [50]
    max_depth: [null]
    min_samples_split: [2]
output_dir: {out}
save_predictions_sample_rows: 5
metrics_average: weighted
fail_on_validation_errors: true
""".strip().format(out=str(tmp_path / "artifacts"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    result_code = run_pipeline(str(cfg_path))
    assert result_code == 0

    expected = [
        "agribot_model.pkl",
        "metrics.json",
        "metrics.md",
        "best_params.json",
        "run_summary.json",
        "predictions_sample.csv",
        "data_validation_report.json",
    ]
    for name in expected:
        assert (tmp_path / "artifacts" / name).exists(), f"missing {name}"
