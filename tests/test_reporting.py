"""Phase 1 report summary."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from sparkles.config.schema import ExperimentConfig, FeatureConfig, TrainConfig
from sparkles.reporting.summary import run_phase1_report


def test_run_phase1_report_shows_latest_metrics(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        symbol="ZZ",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 30),
        train_start=date(2024, 1, 2),
        train_end=date(2024, 3, 1),
        val_start=date(2024, 3, 2),
        val_end=date(2024, 6, 1),
        features=FeatureConfig(log1p_volume=False),
        train=TrainConfig(experiment_name="unit", notes="report test"),
    )
    sym = tmp_path / "artifacts" / "ZZ"
    sym.mkdir(parents=True)
    run_dir = sym / "20990101T000000_000000Z"
    run_dir.mkdir(parents=True)
    metrics = {
        "model_type": "logistic_regression",
        "train_accuracy": 0.5,
        "val_accuracy": 0.4,
        "train_n": 10,
        "val_n": 5,
        "classes": ["take_profit", "stop_loss"],
        "features": {
            "log_entry_close": True,
            "label_geometry": True,
            "intraday_range_pct": True,
            "log1p_volume": False,
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    text = run_phase1_report(cfg, base_dir=tmp_path)
    assert "20990101T000000_000000Z" in text
    assert "val_acc=0.4" in text or "val_acc=0.4 " in text
    assert "train_n=10" in text
    assert "parameters (current experiment YAML)" in text
    assert "splits: train=2024-01-02..2024-03-01" in text
    assert "experiment_name='unit'" in text
    assert '"log1p_volume":false' in text
    assert "features (stored in metrics):" in text
    assert "take_profit" in text
    assert "label_entry_stride=" in text
    assert "ingest_chunk_calendar_days=" in text
    assert "model_type (stored): logistic_regression" in text


def test_run_phase1_report_experiments_tail_shows_model_and_features(
    tmp_path: Path,
) -> None:
    cfg = ExperimentConfig(
        symbol="AB",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 30),
    )
    sym = tmp_path / "artifacts" / "AB"
    sym.mkdir(parents=True)
    run_dir = sym / "run_x"
    run_dir.mkdir(parents=True)
    metrics = {
        "train_accuracy": 1.0,
        "val_accuracy": 0.8,
        "train_n": 3,
        "val_n": 2,
        "classes": ["a"],
        "features": {"log_entry_close": True, "label_geometry": False},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    log = tmp_path / "artifacts" / "experiments.jsonl"
    row = {
        "symbol": "AB",
        "run_id": "run_x",
        "val_accuracy": 0.8,
        "model_type": "logistic_regression",
        "model_solver": "saga",
        "model_class_weight": None,
        "features": {"log_entry_close": True, "label_geometry": False},
        "train_experiment_name": "smoke",
        "train_notes": None,
    }
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = run_phase1_report(cfg, base_dir=tmp_path)
    assert "run_x" in text
    assert "model=logistic_regression/saga" in text
    assert "experiment_name='smoke'" in text


def test_run_phase1_report_explicit_run_id(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        symbol="ZZ",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 30),
    )
    sym = tmp_path / "artifacts" / "ZZ"
    sym.mkdir(parents=True)
    (sym / "run_a").mkdir()
    (sym / "run_b").mkdir()
    m = {"train_accuracy": 1.0, "val_accuracy": 0.9, "train_n": 1, "val_n": 1}
    (sym / "run_b" / "metrics.json").write_text(
        json.dumps(m),
        encoding="utf-8",
    )
    text = run_phase1_report(cfg, base_dir=tmp_path, run_id="run_b")
    assert "run_b" in text
    assert "val_acc=0.9" in text
