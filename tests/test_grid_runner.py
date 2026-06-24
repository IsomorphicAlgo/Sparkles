"""Grid runner file output and progress helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from sparkles.config.grid import build_grid_configs
from sparkles.config.grid_runner import (
    default_progress,
    new_grid_run_dir,
    run_grid_dry_run,
    write_grid_meta,
)
from sparkles.config.schema import ExperimentConfig
from sparkles.models.train import TrainDryRunReport


def test_new_grid_run_dir(tmp_path: Path) -> None:
    d1 = new_grid_run_dir(tmp_path, prefix="test-grid")
    d2 = new_grid_run_dir(tmp_path, prefix="test-grid")
    assert d1.is_dir()
    assert d2.is_dir()
    assert d1 != d2


def test_write_grid_meta(tmp_path: Path) -> None:
    run_dir = new_grid_run_dir(tmp_path, prefix="meta")
    path = write_grid_meta(run_dir, {"n_combinations": 3})
    assert path.is_file()
    assert "n_combinations" in path.read_text(encoding="utf-8")


def test_run_grid_dry_run_writes_files(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.dump(
            {
                "symbol": "RKLB",
                "data_start": date(2024, 1, 1),
                "data_end": date(2024, 6, 1),
                "model": {"type": "logistic_regression"},
                "features": {"log_entry_close": True},
            },
        ),
        encoding="utf-8",
    )
    spec = {
        "experiment_name_prefix": "t",
        "params": {"model.class_weight": ["balanced"]},
    }
    pairs = build_grid_configs(spec, base_path=base)
    run_dir = new_grid_run_dir(tmp_path / "grid_search", prefix="dry")

    fake_report = TrainDryRunReport(
        symbol="RKLB",
        model_type="logistic_regression",
        train_n=10,
        val_n=5,
        train_class_balance={"vertical": 10},
        val_class_balance={"vertical": 5},
        feature_columns=["log_entry_close"],
        features_enabled={"log_entry_close": True},
        experiment_name="t_x",
        notes=None,
        val_rows_dropped_unseen=0,
        sample_weight_method="none",
        ready=True,
    )

    with patch(
        "sparkles.config.grid_runner.dry_run_train",
        return_value=fake_report,
    ):
        df, ready_n = run_grid_dry_run(
            pairs,
            run_dir,
            progress_every=1,
            progress=None,
        )

    assert ready_n == 1
    assert (run_dir / "dry_run_log.txt").is_file()
    assert (run_dir / "dry_run_summary.csv").is_file()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_default_progress(capsys) -> None:
    default_progress(50, 100, "dry-run")
    out = capsys.readouterr().out
    assert "[50/100]" in out
