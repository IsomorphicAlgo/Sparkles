"""Phase G2 session / volume / VWAP feature tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.dataset import build_feature_matrix, feature_warmup_bars
from sparkles.features.session import build_session_time, build_vwap_distance
from tests.test_dataset import _cfg


def _synthetic_session_ohlcv() -> pd.DataFrame:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=120, freq="1min", tz=tz)
    close = pd.Series(100.0 + np.arange(120, dtype=float) * 0.01, index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000 + np.arange(120) * 10,
        },
        index=idx,
    )


def test_session_time_at_open_and_midday() -> None:
    ohlcv = _synthetic_session_ohlcv()
    labels = pd.DataFrame(
        {
            "entry_close": [ohlcv["close"].iloc[0], ohlcv["close"].iloc[60]],
            "barrier_outcome": ["vertical", "vertical"],
            "sigma_ann_at_entry": [0.5, 0.5],
            "vol_scale_ratio": [1.0, 1.0],
            "tp_move_effective": [0.1, 0.1],
            "sl_move": [0.05, 0.05],
        },
        index=ohlcv.index[[0, 60]],
    )
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        session_time=True,
    )
    ctx = EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=ohlcv.reindex(labels.index),
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=pd.Series([0, 60], index=labels.index),
        feature_config=fc,
        exchange_timezone="America/New_York",
    )
    X = build_session_time(ctx)
    assert X.loc[labels.index[0], "minutes_since_open"] == pytest.approx(0.0)
    assert X.loc[labels.index[0], "minutes_to_close"] == pytest.approx(119.0)
    assert X.loc[labels.index[1], "minutes_since_open"] == pytest.approx(60.0)
    assert math.isfinite(float(X.loc[labels.index[1], "sin_time"]))


def test_vwap_distance_zero_at_session_open() -> None:
    ohlcv = _synthetic_session_ohlcv()
    labels = pd.DataFrame(
        {
            "entry_close": [ohlcv["close"].iloc[0]],
            "barrier_outcome": ["vertical"],
            "sigma_ann_at_entry": [0.5],
            "vol_scale_ratio": [1.0],
            "tp_move_effective": [0.1],
            "sl_move": [0.05],
        },
        index=ohlcv.index[[0]],
    )
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        vwap_distance=True,
    )
    ctx = EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=ohlcv.reindex(labels.index),
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=pd.Series([0], index=labels.index),
        feature_config=fc,
        exchange_timezone="America/New_York",
    )
    X = build_vwap_distance(ctx)
    assert X["vwap_session_dist_pct"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_g2_columns_when_all_enabled() -> None:
    ohlcv = _synthetic_session_ohlcv()
    labels = pd.DataFrame(
        {
            "entry_close": ohlcv["close"].iloc[80:],
            "barrier_outcome": ["vertical"] * 40,
            "sigma_ann_at_entry": [0.5] * 40,
            "vol_scale_ratio": [1.0] * 40,
            "tp_move_effective": [0.1] * 40,
            "sl_move": [0.05] * 40,
        },
        index=ohlcv.index[80:],
    )
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        session_time=True,
        volume_context=True,
        volume_median_window_bars=30,
        vwap_distance=True,
    )
    X, _y = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    assert list(X.columns) == [
        "minutes_since_open",
        "minutes_to_close",
        "sin_time",
        "cos_time",
        "rel_volume",
        "log_rel_volume",
        "vwap_session_dist_pct",
    ]
    assert not X.isna().any().any()


def test_feature_warmup_includes_volume_window() -> None:
    fc = FeatureConfig(
        log_entry_close=True,
        volume_context=True,
        volume_median_window_bars=45,
    )
    assert feature_warmup_bars(fc) == 45
