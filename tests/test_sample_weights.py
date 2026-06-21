"""Phase I4 sample weights (label uniqueness)."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import ExperimentConfig
from sparkles.models.sample_weights import (
    resolve_fit_sample_weights,
    uniqueness_weights,
)
from sparkles.models.train import run_train


def test_uniqueness_downweights_overlapping_windows() -> None:
    positions = np.array([10, 10], dtype=np.float64)
    forward = np.array([5, 5], dtype=np.int64)
    w = uniqueness_weights(positions, forward)
    assert w[0] == pytest.approx(0.5)
    assert w[1] == pytest.approx(0.5)


def test_uniqueness_non_overlapping_is_one() -> None:
    positions = np.array([10, 20], dtype=np.float64)
    forward = np.array([3, 3], dtype=np.int64)
    w = uniqueness_weights(positions, forward)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(1.0)


def test_resolve_fit_sample_weights_none_without_class_weight() -> None:
    cfg = ExperimentConfig(
        symbol="T",
        data_start=date(2024, 6, 1),
        data_end=date(2024, 6, 10),
        train={"sample_weight_method": "none"},
        model={"class_weight": None},
    )
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = np.array([0, 1, 0])
    le.fit(["stop_loss", "take_profit"])
    labels = pd.DataFrame(
        {"bars_forward": [2, 2, 2]},
        index=pd.DatetimeIndex(
            [
                "2024-06-03 10:00",
                "2024-06-03 11:00",
                "2024-06-03 12:00",
            ],
        ),
    )
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=10, freq="1min", tz=tz)
    ohlcv = pd.DataFrame({"close": 1.0}, index=idx)
    sw, summary = resolve_fit_sample_weights(
        cfg,
        le,
        y,
        labels.index,
        labels,
        ohlcv,
    )
    assert sw is None
    assert summary["sample_weight_method"] == "none"


def test_run_train_records_uniqueness_method(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 10:01", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
        ],
    )
    labels = pd.DataFrame(
        {
            "entry_close": 100.0,
            "barrier_outcome": [
                "take_profit",
                "stop_loss",
                "take_profit",
                "stop_loss",
                "take_profit",
                "vertical",
            ],
            "tp_move_effective": 0.04,
            "sl_move": 0.02,
            "bars_forward": [3, 3, 3, 3, 3, 3],
            "sigma_ann_at_entry": 0.2,
            "vol_scale_ratio": 1.0,
        },
        index=idx,
    )
    bar_idx = pd.date_range("2024-06-03 09:30", "2024-06-06 12:00", freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        },
        index=bar_idx,
    )
    cfg = ExperimentConfig(
        symbol="T",
        exchange_timezone="America/New_York",
        data_start=date(2024, 6, 1),
        data_end=date(2024, 6, 10),
        train_start=date(2024, 6, 3),
        train_end=date(2024, 6, 4),
        val_start=date(2024, 6, 5),
        val_end=date(2024, 6, 7),
        features={
            "log_entry_close": True,
            "label_geometry": True,
            "intraday_range_pct": True,
            "log1p_volume": True,
        },
        train={"sample_weight_method": "uniqueness", "export_predictions": "none"},
        paths={"cache_dir": "data/cache", "artifacts_dir": "artifacts"},
    )
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sample_weight_method"] == "uniqueness"
    assert metrics["sample_weight_mean"] is not None
