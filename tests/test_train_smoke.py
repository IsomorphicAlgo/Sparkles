"""End-to-end train with injected frames (no large Parquet fixtures)."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from sparkles.config.schema import ExperimentConfig, ModelConfig, TrainConfig
from sparkles.models.train import run_train


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
        "train_start": date(2024, 6, 3),
        "train_end": date(2024, 6, 4),
        "val_start": date(2024, 6, 5),
        "val_end": date(2024, 6, 7),
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_run_train_writes_artifacts(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg()
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    assert out.is_dir()
    assert (out / "model_bundle.joblib").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "predictions.parquet").is_file()
    log = tmp_path / "artifacts" / "experiments.jsonl"
    assert log.is_file()
    last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["model_type"] == "logistic_regression"
    assert last["model_solver"] == "lbfgs"
    assert last["train_drop_val_unseen_classes"] is True
    assert last["features"]["log_entry_close"] is True


def test_run_train_xgboost_writes_artifacts(tmp_path) -> None:
    pytest.importorskip("xgboost")
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg(
        model=ModelConfig(
            type="xgboost_classifier",
            xgb_n_estimators=8,
            xgb_max_depth=2,
            xgb_learning_rate=0.2,
        ),
    )
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["model_type"] == "xgboost_classifier"
    assert m["val_accuracy"] is not None


def test_min_train_rows_blocks(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 2,
            "high": [101.0] * 2,
            "low": [99.0] * 2,
            "close": [100.0] * 2,
            "volume": [1e6] * 2,
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 2,
            "barrier_outcome": ["take_profit", "take_profit"],
            "sigma_ann_at_entry": [0.6] * 2,
            "vol_scale_ratio": [1.0] * 2,
            "tp_move_effective": [0.1] * 2,
            "sl_move": [0.05] * 2,
        },
        index=idx,
    )
    cfg = _cfg(train=TrainConfig(min_train_rows=5))
    with pytest.raises(ValueError, match="min_train_rows"):
        run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)


def test_experiment_name_logged(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg(
        train=TrainConfig(
            experiment_name="smoke-test",
            notes="phase-a",
        ),
    )
    run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    log = tmp_path / "artifacts" / "experiments.jsonl"
    last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["train_experiment_name"] == "smoke-test"
    assert last["train_notes"] == "phase-a"


def test_drop_val_unseen_classes_false_raises(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.0] * 3,
            "volume": [1e6] * 3,
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 3,
            "barrier_outcome": ["take_profit", "stop_loss", "end_of_data"],
            "sigma_ann_at_entry": [0.6] * 3,
            "vol_scale_ratio": [1.0] * 3,
            "tp_move_effective": [0.1] * 3,
            "sl_move": [0.05] * 3,
        },
        index=idx,
    )
    cfg = _cfg(train=TrainConfig(drop_val_unseen_classes=False))
    with pytest.raises(ValueError, match="unseen"):
        run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
