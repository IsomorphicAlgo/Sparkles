"""Training dataset: entry-only features and session-date splits."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sparkles.config.schema import ExperimentConfig
from sparkles.features.dataset import (
    build_feature_matrix,
    train_val_masks_by_session_date,
)


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


def test_build_feature_matrix_aligns_to_ohlcv() -> None:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=4, freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.2, 10.1, 10.3],
            "low": [9.9, 9.95, 9.98, 10.0],
            "close": [10.1, 10.0, 10.05, 10.1],
            "volume": [1000, 1100, 900, 1000],
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "entry_close": [10.1, 10.0, 10.05, 10.1],
            "barrier_outcome": ["take_profit"] * 4,
            "sigma_ann_at_entry": [0.5] * 4,
            "vol_scale_ratio": [1.0] * 4,
            "tp_move_effective": [0.1] * 4,
            "sl_move": [0.05] * 4,
        },
        index=idx,
    )
    X, y = build_feature_matrix(labels, ohlcv, _cfg())
    assert len(X) == 4
    assert list(X.columns) == [
        "log_entry_close",
        "sigma_ann_at_entry",
        "vol_scale_ratio",
        "tp_move_effective",
        "sl_move",
        "intraday_range_pct",
        "log1p_volume",
    ]
    assert y.iloc[0] == "take_profit"


def test_build_feature_matrix_drops_unmatched_timestamps() -> None:
    tz = "America/New_York"
    idx_l = pd.date_range("2024-06-03 09:30", periods=2, freq="1min", tz=tz)
    idx_o = pd.date_range("2024-06-03 09:31", periods=1, freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.0],
            "volume": [100],
        },
        index=idx_o,
    )
    labels = pd.DataFrame(
        {
            "entry_close": [10.0, 10.0],
            "barrier_outcome": ["vertical", "take_profit"],
            "sigma_ann_at_entry": [0.5, 0.5],
            "vol_scale_ratio": [1.0, 1.0],
            "tp_move_effective": [0.1, 0.1],
            "sl_move": [0.05, 0.05],
        },
        index=idx_l,
    )
    X, _y = build_feature_matrix(labels, ohlcv, _cfg())
    assert len(X) == 1


def test_train_val_masks_require_yaml_dates() -> None:
    cfg = ExperimentConfig(
        symbol="X",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 12, 31),
    )
    idx = pd.DatetimeIndex(["2024-06-03 09:30:00"], tz="America/New_York")
    try:
        train_val_masks_by_session_date(idx, cfg)
    except ValueError as e:
        assert "train_start" in str(e)
    else:
        raise AssertionError("expected ValueError")
