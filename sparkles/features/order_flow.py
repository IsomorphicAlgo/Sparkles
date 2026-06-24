"""OHLCV-only order-flow and liquidity proxies (ML expansion Phase G4c).

No Level-2 data — spread/illiquidity estimates are literature-backed proxies only.
All trailing series use bars at or before each entry timestamp (no lookahead).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.intraday import _slice_at_entries

_CS_K = 3.0 - 2.0 * math.sqrt(2.0)


def _corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling intraday Corwin–Schultz (2012) spread estimator."""
    lo = low.astype(np.float64).clip(lower=1e-12)
    hi = high.astype(np.float64)
    log_hl_sq = np.log(hi / lo) ** 2
    beta = (
        log_hl_sq.rolling(2, min_periods=2)
        .sum()
        .rolling(window, min_periods=window)
        .mean()
    )
    h2 = hi.rolling(2, min_periods=2).max()
    l2 = lo.rolling(2, min_periods=2).min()
    gamma = np.log(h2 / l2.clip(lower=1e-12)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta.clip(lower=0.0))) / _CS_K - np.sqrt(
        (gamma / _CS_K).clip(lower=0.0),
    )
    alpha = alpha.clip(lower=0.0)
    exp_a = np.exp(alpha)
    return 2.0 * (exp_a - 1.0) / (1.0 + exp_a)


def _roll_implied_spread(close: pd.Series, window: int) -> pd.Series:
    """Roll (1984) spread from first-order return autocovariance."""
    c = close.astype(np.float64)
    ret = pd.Series(np.log((c / c.shift(1)).to_numpy(dtype=float)), index=c.index)
    lag = ret.shift(1)
    mean_r = ret.rolling(window, min_periods=window).mean()
    mean_lag = lag.rolling(window, min_periods=window).mean()
    mean_prod = (ret * lag).rolling(window, min_periods=window).mean()
    cov = mean_prod - mean_r * mean_lag
    neg = (-cov).clip(lower=0.0)
    return 2.0 * np.sqrt(neg)


def _amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Amihud (2002) illiquidity: mean |r| / dollar_volume over trailing window."""
    c = close.astype(np.float64)
    vol = volume.astype(np.float64)
    ret_abs = np.abs(np.log((c / c.shift(1)).to_numpy(dtype=float)))
    ret_abs = pd.Series(ret_abs, index=c.index)
    dollar_vol = (c * vol).replace(0.0, np.nan)
    illiq_bar = ret_abs / dollar_vol
    return illiq_bar.rolling(window, min_periods=window).mean()


def _entry_wick_pcts(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    span = (high.astype(np.float64) - low.astype(np.float64)).clip(lower=1e-12)
    op = open_.astype(np.float64)
    cl = close.astype(np.float64)
    hi = high.astype(np.float64)
    lo = low.astype(np.float64)
    body_top = np.maximum(op, cl)
    body_bot = np.minimum(op, cl)
    upper = (hi - body_top) / span
    lower = (body_bot - lo) / span
    return (
        pd.Series(upper, index=close.index, dtype=float),
        pd.Series(lower, index=close.index, dtype=float),
    )


def build_order_flow_proxies(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    ohlcv = ctx.full_ohlcv
    aligned = ctx.aligned_ohlcv
    pos = ctx.entry_positions
    roll_w = fc.roll_window_bars
    amihud_w = fc.amihud_window_bars

    cs = _corwin_schultz_spread(ohlcv["high"], ohlcv["low"], roll_w)
    roll_sp = _roll_implied_spread(ohlcv["close"], roll_w)
    amihud = _amihud_illiquidity(ohlcv["close"], ohlcv["volume"], amihud_w)

    vol = ohlcv["volume"].astype(np.float64)
    vol_med = vol.rolling(roll_w, min_periods=roll_w).median()
    rel_vol = vol / vol_med.clip(lower=1e-12)
    span = (aligned["high"].astype(np.float64) - aligned["low"].astype(np.float64)).clip(
        lower=1e-12,
    )
    body_dir = (aligned["close"].astype(np.float64) - aligned["open"].astype(np.float64)) / span
    rel_vol_at_entry = _slice_at_entries(rel_vol, pos)
    signed_vol = body_dir * rel_vol_at_entry

    upper, lower = _entry_wick_pcts(
        aligned["open"],
        aligned["high"],
        aligned["low"],
        aligned["close"],
    )

    return pd.DataFrame(
        {
            "corwin_schultz_spread": _slice_at_entries(cs, pos),
            "roll_implied_spread": _slice_at_entries(roll_sp, pos),
            "amihud_illiq": _slice_at_entries(amihud, pos),
            "signed_volume_proxy": signed_vol,
            "upper_wick_pct": upper,
            "lower_wick_pct": lower,
        },
        index=pos.index,
    )


def g4c_warmup_bars(fc: FeatureConfig) -> int:
    if not fc.order_flow_proxies:
        return 0
    return max(fc.roll_window_bars, fc.amihud_window_bars)
