"""Phase G1 intraday trailing-window feature tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import FeatureConfig
from sparkles.features.dataset import build_feature_matrix
from sparkles.features.intraday import max_warmup_bars
from tests.test_dataset import _cfg


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=n, freq="1min", tz=tz)
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.01, index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000 + np.arange(n),
        },
        index=idx,
    )


def _labels_for_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {
            "entry_close": np.linspace(100.0, 101.0, n),
            "barrier_outcome": ["take_profit"] * n,
            "sigma_ann_at_entry": [0.5] * n,
            "vol_scale_ratio": [1.0] * n,
            "tp_move_effective": [0.1] * n,
            "sl_move": [0.05] * n,
        },
        index=idx,
    )


def test_max_warmup_bars_g1() -> None:
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        returns_multi_horizon=True,
        realized_vol_multi=True,
        range_vol_multi=True,
    )
    assert max_warmup_bars(fc) == 120


def test_g1_return_column_values() -> None:
    ohlcv = _synthetic_ohlcv(200)
    # Entry at bar 60: ret_5m = log(close[60]/close[55])
    entry_ix = ohlcv.index[60:61]
    labels = _labels_for_index(entry_ix)
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        returns_multi_horizon=True,
        returns_horizons_bars=[5],
    )
    X, _y = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    expected = math.log(float(ohlcv["close"].iloc[60] / ohlcv["close"].iloc[55]))
    assert len(X) == 1
    assert X["ret_5m"].iloc[0] == pytest.approx(expected)


def test_g1_columns_when_all_enabled() -> None:
    ohlcv = _synthetic_ohlcv(250)
    labels = _labels_for_index(ohlcv.index[150:])
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        returns_multi_horizon=True,
        realized_vol_multi=True,
        range_vol_multi=True,
    )
    X, _y = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    assert list(X.columns) == [
        "ret_5m",
        "ret_15m",
        "ret_30m",
        "ret_60m",
        "rv_30m",
        "rv_120m",
        "rv_ratio_30_120m",
        "parkinson_30m",
        "atr_norm_30m",
    ]
    assert not X.isna().any().any()


def test_g1_drops_insufficient_warmup() -> None:
    ohlcv = _synthetic_ohlcv(200)
    labels = _labels_for_index(ohlcv.index[:130])
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        returns_multi_horizon=True,
        realized_vol_multi=True,
        range_vol_multi=True,
    )
    X, _y = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    # Warm-up 120 bars → keep positions >= 120 → 130 - 120 = 10 rows
    assert len(X) == 10


def test_feature_config_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="returns_horizons_bars"):
        FeatureConfig(
            log_entry_close=True,
            returns_multi_horizon=True,
            returns_horizons_bars=[],
        )
