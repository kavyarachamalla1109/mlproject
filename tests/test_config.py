from src.config import load_config


def test_load_config_has_required_keys() -> None:
    cfg = load_config("configs/train_config.yaml")
    assert cfg.data_path.endswith("crop_recommendation_sample.csv")
    assert cfg.target_column == "label"
    assert cfg.tuning.enabled is True
    assert "n_estimators" in cfg.tuning.param_grid
