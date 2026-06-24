"""Phase G4c order-flow proxy feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.dataset import build_feature_matrix
from sparkles.features.order_flow import build_order_flow_proxies, g4c_warmup_bars
from tests.test_dataset import _cfg


def _synthetic_ohlcv(n: int = 80, *, wide_range: bool = False) -> pd.DataFrame:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=n, freq="1min", tz=tz)
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.02, index=idx)
    hl = 0.5 if wide_range else 0.05
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + hl,
            "low": close - hl,
            "close": close,
            "volume": 1000.0 + np.arange(n, dtype=float) * 10.0,
        },
        index=idx,
    )


def _ctx_at_bar(ohlcv: pd.DataFrame, bar: int, fc: FeatureConfig) -> EntryFeatureContext:
    entry_ix = ohlcv.index[bar : bar + 1]
    labels = pd.DataFrame(
        {
            "entry_close": [float(ohlcv["close"].iloc[bar])],
            "barrier_outcome": ["vertical"],
            "sigma_ann_at_entry": [0.5],
            "vol_scale_ratio": [1.0],
            "tp_move_effective": [0.1],
            "sl_move": [0.05],
        },
        index=entry_ix,
    )
    return EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=ohlcv.reindex(entry_ix),
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=pd.Series([bar], index=entry_ix, dtype=np.int64),
        feature_config=fc,
        exchange_timezone="America/New_York",
    )


def _g4c_fc(**kwargs: object) -> FeatureConfig:
    base = {
        "log_entry_close": False,
        "label_geometry": False,
        "intraday_range_pct": False,
        "log1p_volume": False,
        "order_flow_proxies": True,
        "roll_window_bars": 20,
        "amihud_window_bars": 20,
    }
    base.update(kwargs)
    return FeatureConfig(**base)


def test_g4c_warmup_bars() -> None:
    fc = _g4c_fc(roll_window_bars=15, amihud_window_bars=25)
    assert g4c_warmup_bars(fc) == 25
    off = FeatureConfig(log_entry_close=True, order_flow_proxies=False)
    assert g4c_warmup_bars(off) == 0


def test_wide_range_corwin_schultz_higher_than_tight() -> None:
    fc = _g4c_fc()
    wide = _synthetic_ohlcv(80, wide_range=True)
    tight = _synthetic_ohlcv(80, wide_range=False)
    bar = 79
    cs_wide = build_order_flow_proxies(_ctx_at_bar(wide, bar, fc))[
        "corwin_schultz_spread"
    ].iloc[0]
    cs_tight = build_order_flow_proxies(_ctx_at_bar(tight, bar, fc))[
        "corwin_schultz_spread"
    ].iloc[0]
    assert cs_wide > cs_tight


def test_wick_pcts_bullish_bar() -> None:
    tz = "America/New_York"
    ts = pd.Timestamp("2024-06-03 10:00", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.8],
            "volume": [5000.0],
        },
        index=[ts],
    )
    fc = _g4c_fc(roll_window_bars=2, amihud_window_bars=2)
    X = build_order_flow_proxies(_ctx_at_bar(ohlcv, 0, fc))
    assert X["upper_wick_pct"].iloc[0] == pytest.approx(0.1, abs=1e-9)
    assert X["lower_wick_pct"].iloc[0] == pytest.approx(0.5, abs=1e-9)


def test_zero_volume_amihud_nan() -> None:
    ohlcv = _synthetic_ohlcv(40)
    ohlcv.loc[ohlcv.index[-1], "volume"] = 0.0
    fc = _g4c_fc(roll_window_bars=5, amihud_window_bars=5)
    X = build_order_flow_proxies(_ctx_at_bar(ohlcv, len(ohlcv) - 1, fc))
    assert np.isnan(float(X["amihud_illiq"].iloc[0]))


def test_g4c_columns_in_feature_matrix() -> None:
    ohlcv = _synthetic_ohlcv(60)
    labels = pd.DataFrame(
        {
            "entry_close": ohlcv["close"].iloc[40:],
            "barrier_outcome": ["vertical"] * 20,
            "sigma_ann_at_entry": [0.5] * 20,
            "vol_scale_ratio": [1.0] * 20,
            "tp_move_effective": [0.1] * 20,
            "sl_move": [0.05] * 20,
        },
        index=ohlcv.index[40:],
    )
    fc = _g4c_fc(roll_window_bars=10, amihud_window_bars=10)
    X, _y = build_feature_matrix(labels, ohlcv, _cfg(features=fc))
    assert list(X.columns) == [
        "corwin_schultz_spread",
        "roll_implied_spread",
        "amihud_illiq",
        "signed_volume_proxy",
        "upper_wick_pct",
        "lower_wick_pct",
    ]
    assert not X.isna().any().any()
