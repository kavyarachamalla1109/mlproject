from pathlib import Path

from src.config import load_config
from src.validate import validate_data_and_config


def test_validate_sample_data_passes() -> None:
    cfg = load_config("configs/train_config.yaml")
    report = validate_data_and_config(cfg, cfg.output_dir)
    assert report["validation_passed"] is True
    assert report["rows"] >= 30
    assert Path(cfg.output_dir, "data_validation_report.json").exists()
