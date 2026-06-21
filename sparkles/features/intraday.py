"""Trailing intraday features from full 1m OHLCV (ML expansion Phase G1).

All series use only bars at or before each entry timestamp (no lookahead).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext

_LN2 = math.log(2.0)
_PARKINSON_SCALE = 1.0 / (4.0 * _LN2)


def _one_minute_log_returns(close: pd.Series) -> pd.Series:
    c = close.astype(np.float64)
    ratio = c / c.shift(1)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=c.index, dtype=float)


def _slice_at_entries(series: pd.Series, positions: pd.Series) -> pd.Series:
    """Map a full-bar series to label rows via integer positions."""
    pos = positions.to_numpy(dtype=np.int64, copy=False)
    vals = series.to_numpy(dtype=float, copy=False)
    out = vals[pos]
    return pd.Series(out, index=positions.index, dtype=float)


def _horizon_column_name(bars: int) -> str:
    return f"ret_{bars}m"


def _rv_column_name(bars: int) -> str:
    return f"rv_{bars}m"


def _parkinson_column_name(bars: int) -> str:
    return f"parkinson_{bars}m"


def _atr_norm_column_name(bars: int) -> str:
    return f"atr_norm_{bars}m"


def build_returns_multi_horizon(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    close = ctx.full_ohlcv["close"]
    pos = ctx.entry_positions
    cols: dict[str, pd.Series] = {}
    for h in fc.returns_horizons_bars:
        lagged = close.astype(np.float64) / close.astype(np.float64).shift(h)
        ret = pd.Series(np.log(lagged.to_numpy(dtype=float)), index=close.index, dtype=float)
        cols[_horizon_column_name(h)] = _slice_at_entries(ret, pos)
    return pd.DataFrame(cols, index=pos.index)


def build_realized_vol_multi(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    close = ctx.full_ohlcv["close"]
    pos = ctx.entry_positions
    log_ret = _one_minute_log_returns(close)
    cols: dict[str, pd.Series] = {}
    rv_at_entry: dict[int, pd.Series] = {}
    for w in fc.realized_vol_windows_bars:
        rv = log_ret.rolling(window=w, min_periods=w).std()
        sliced = _slice_at_entries(rv, pos)
        cols[_rv_column_name(w)] = sliced
        rv_at_entry[w] = sliced
    if fc.realized_vol_include_ratio and len(fc.realized_vol_windows_bars) >= 2:
        short_w = min(fc.realized_vol_windows_bars)
        long_w = max(fc.realized_vol_windows_bars)
        ratio = rv_at_entry[short_w] / rv_at_entry[long_w].clip(lower=1e-12)
        cols[f"rv_ratio_{short_w}_{long_w}m"] = ratio
    return pd.DataFrame(cols, index=pos.index)


def build_range_vol_multi(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    ohlcv = ctx.full_ohlcv
    pos = ctx.entry_positions
    w = fc.range_vol_window_bars
    close = ohlcv["close"].astype(np.float64)
    hi = ohlcv["high"].astype(np.float64)
    lo = ohlcv["low"].astype(np.float64)

    log_hl = np.log((hi / lo.clip(lower=1e-12)).to_numpy(dtype=float))
    park_bar = pd.Series(log_hl * log_hl * _PARKINSON_SCALE, index=ohlcv.index, dtype=float)
    parkinson = np.sqrt(park_bar.rolling(window=w, min_periods=w).mean())

    cols: dict[str, pd.Series] = {
        _parkinson_column_name(w): _slice_at_entries(parkinson, pos),
    }

    if fc.range_vol_include_atr_norm:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                hi - lo,
                (hi - prev_close).abs(),
                (lo - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=w, min_periods=w).mean()
        atr_norm = atr / close.clip(lower=1e-12)
        cols[_atr_norm_column_name(w)] = _slice_at_entries(atr_norm, pos)

    return pd.DataFrame(cols, index=pos.index)


def max_warmup_bars(fc: FeatureConfig) -> int:
    """Minimum full-OHLCV bars before an entry row can have complete G1 features."""
    need = 0
    if fc.returns_multi_horizon and fc.returns_horizons_bars:
        need = max(need, max(fc.returns_horizons_bars))
    if fc.realized_vol_multi and fc.realized_vol_windows_bars:
        need = max(need, max(fc.realized_vol_windows_bars))
    if fc.range_vol_multi:
        need = max(need, fc.range_vol_window_bars)
    return need
