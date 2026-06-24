"""Phase G4a technical indicator feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import FeatureConfig
from sparkles.features.dataset import build_feature_matrix
from sparkles.features.registry import assemble_feature_columns
from sparkles.features.technical import build_technical_indicators, g4a_warmup_bars
from sparkles.features.builders import EntryFeatureContext
from tests.test_dataset import _cfg


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=n, freq="1min", tz=tz)
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.05, index=idx)
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
            "entry_close": np.linspace(100.0, 110.0, n),
            "barrier_outcome": ["take_profit"] * n,
            "sigma_ann_at_entry": [0.5] * n,
            "vol_scale_ratio": [1.0] * n,
            "tp_move_effective": [0.1] * n,
            "sl_move": [0.05] * n,
        },
        index=idx,
    )


def _ctx_at_bar(ohlcv: pd.DataFrame, bar: int, fc: FeatureConfig) -> EntryFeatureContext:
    entry_ix = ohlcv.index[bar : bar + 1]
    labels = _labels_for_index(entry_ix)
    positions = pd.Series([bar], index=entry_ix, dtype=np.int64)
    return EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=ohlcv.reindex(entry_ix),
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=positions,
        feature_config=fc,
        exchange_timezone="America/New_York",
    )


def test_g4a_warmup_bars() -> None:
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        technical_indicators=True,
        ema_windows_bars=[9, 21, 50],
        macd_slow_bars=26,
        macd_signal_bars=9,
    )
    assert g4a_warmup_bars(fc) == 50


def test_g4a_monotone_uptrend_rsi_high() -> None:
    ohlcv = _synthetic_ohlcv(200)
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        technical_indicators=True,
        ema_windows_bars=[9],
        rsi_window_bars=14,
    )
    ctx = _ctx_at_bar(ohlcv, 100, fc)
    X = build_technical_indicators(ctx)
    assert X["rsi_14m"].iloc[0] > 0.7


def test_g4a_ema_dist_positive_in_uptrend() -> None:
    ohlcv = _synthetic_ohlcv(200)
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        technical_indicators=True,
        ema_windows_bars=[9],
        rsi_window_bars=14,
    )
    ctx = _ctx_at_bar(ohlcv, 150, fc)
    X = build_technical_indicators(ctx)
    assert X["ema_dist_9m"].iloc[0] > 0.0


def test_g4a_columns_when_enabled() -> None:
    ohlcv = _synthetic_ohlcv(200)
    entry_ix = ohlcv.index[100:101]
    labels = _labels_for_index(entry_ix)
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        technical_indicators=True,
        ema_windows_bars=[9, 21],
    )
    X, _ = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    assert "ema_dist_9m" in X.columns
    assert "ema_dist_21m" in X.columns
    assert "rsi_14m" in X.columns
    assert "macd_line" in X.columns
    assert "macd_signal" in X.columns
    assert "macd_hist" in X.columns


def test_g4a_toggle_off_excluded_from_registry() -> None:
    fc = FeatureConfig(
        log_entry_close=True,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        technical_indicators=False,
    )
    ohlcv = _synthetic_ohlcv(50)
    labels = _labels_for_index(ohlcv.index[40:41])
    ctx = _ctx_at_bar(ohlcv, 40, fc)
    X = assemble_feature_columns(ctx, fc)
    assert "rsi_14m" not in X.columns
    assert "macd_line" not in X.columns


def test_macd_fast_must_be_less_than_slow() -> None:
    with pytest.raises(ValueError, match="macd_fast_bars"):
        FeatureConfig(
            log_entry_close=True,
            label_geometry=False,
            intraday_range_pct=False,
            log1p_volume=False,
            macd_fast_bars=26,
            macd_slow_bars=12,
        )
