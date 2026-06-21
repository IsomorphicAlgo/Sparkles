"""Config merge helpers for preset overlays."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from sparkles.config.load import deep_merge_mappings, load_experiment_config_merged
from sparkles.config.schema import ExperimentConfig


def test_deep_merge_nested() -> None:
    base = {"model": {"type": "logistic_regression", "tol": 1e-4}, "symbol": "RKLB"}
    overlay = {"model": {"class_weight": "balanced"}}
    merged = deep_merge_mappings(base, overlay)
    assert merged["symbol"] == "RKLB"
    assert merged["model"]["type"] == "logistic_regression"
    assert merged["model"]["tol"] == 1e-4
    assert merged["model"]["class_weight"] == "balanced"


def test_load_experiment_config_merged(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.dump(
            {
                "symbol": "RKLB",
                "data_start": date(2024, 1, 1),
                "data_end": date(2024, 6, 1),
                "model": {"type": "logistic_regression"},
            },
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.dump(
            {
                "train": {"experiment_name": "merged-test"},
                "model": {"class_weight": "balanced"},
            },
        ),
        encoding="utf-8",
    )
    cfg = load_experiment_config_merged(base, overlay)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.symbol == "RKLB"
    assert cfg.model.class_weight == "balanced"
    assert cfg.train.experiment_name == "merged-test"
