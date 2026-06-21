"""Phase I3 meta-label spike."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sparkles.backtest.meta_label import (
    META_FEATURE_COLUMNS,
    assert_meta_train_within_primary_train,
    build_meta_feature_matrix,
    compare_entry_policies,
    meta_label_targets,
    train_meta_label,
)
from sparkles.config.schema import ExperimentConfig
from sparkles.models.train import prepare_training_data, run_train


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "exchange_timezone": "America/New_York",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
        "train_start": date(2024, 6, 3),
        "train_end": date(2024, 6, 4),
        "val_start": date(2024, 6, 5),
        "val_end": date(2024, 6, 7),
        "label_entry_stride": 1,
        "model": {"type": "logistic_regression", "max_iter": 500},
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_meta_label_targets() -> None:
    y = pd.Series(["take_profit", "stop_loss", "vertical"])
    assert meta_label_targets(y).tolist() == [1, 0, 0]


def test_assert_meta_train_within_primary_train_rejects_val_leakage() -> None:
    tz = "America/New_York"
    tr = pd.DatetimeIndex([pd.Timestamp("2024-06-03 10:00", tz=tz)])
    va = pd.DatetimeIndex([pd.Timestamp("2024-06-05 10:00", tz=tz)])
    with pytest.raises(ValueError, match="leaked"):
        assert_meta_train_within_primary_train(va, tr)


def test_build_meta_feature_matrix() -> None:
    df = pd.DataFrame(
        {
            "proba_stop_loss": [0.1],
            "proba_take_profit": [0.2],
            "proba_vertical": [0.7],
            "max_proba": [0.7],
        },
    )
    x = build_meta_feature_matrix(df)
    assert list(x.columns) == list(META_FEATURE_COLUMNS)


def test_train_and_compare_meta_label(tmp_path: Path) -> None:
    tz = "America/New_York"
    cfg = _cfg(
        paths={"cache_dir": "data/cache", "artifacts_dir": "artifacts"},
        train={"export_predictions": "val"},
        features={
            "log_entry_close": True,
            "label_geometry": False,
            "intraday_range_pct": True,
            "log1p_volume": True,
        },
    )
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)

    rows = []
    for day in (3, 4, 5, 6, 7):
        for hour in (10, 11):
            rows.append(
                {
                    "entry_time": pd.Timestamp(f"2024-06-0{day} {hour:02d}:00", tz=tz),
                    "entry_close": 100.0,
                    "tp_move_effective": 0.04,
                    "sl_move": 0.02,
                    "bars_forward": 1,
                    "barrier_outcome": "take_profit" if hour == 10 else "stop_loss",
                },
            )
    labels = pd.DataFrame(rows).set_index("entry_time")
    labels.to_parquet(cache / "T_labeled_2024-06-01_2024-06-10_s1.parquet")

    idx = pd.date_range("2024-06-03 09:30", "2024-06-07 12:00", freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        },
        index=idx,
    )
    ohlcv.to_parquet(cache / "T_1min_2024-06-01_2024-06-10.parquet")

    run_dir = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    _, metrics = train_meta_label(
        cfg,
        run_dir,
        primary_threshold=0.01,
        base_dir=tmp_path,
    )
    assert metrics["n_meta_train_gated"] >= 1
    prep = prepare_training_data(cfg, base_dir=tmp_path)
    assert metrics["n_meta_train_gated"] <= len(prep.X_tr)

    results = compare_entry_policies(
        cfg,
        run_dir,
        primary_threshold=0.01,
        enforce_day_trade_cap=False,
        base_dir=tmp_path,
    )
    assert "meta_label" in results["policies"]
    assert (run_dir / "meta_label_compare.json").is_file()
