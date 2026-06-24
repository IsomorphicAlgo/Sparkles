"""Session time, volume context, and VWAP distance (ML expansion Phase G2).

All features use only bars at or before each entry timestamp on the full 1m OHLCV
series (exchange timezone from experiment config).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.intraday import _slice_at_entries
from sparkles.features.volatility import ensure_exchange_tz_index

_TWO_PI = 2.0 * math.pi


def _session_dates(index: pd.DatetimeIndex | pd.Index, exchange_timezone: str) -> pd.Series:
    ix = ensure_exchange_tz_index(pd.DatetimeIndex(index), exchange_timezone)
    return pd.Series(pd.DatetimeIndex(ix).normalize().date, index=index, dtype=object)


def build_session_time(ctx: EntryFeatureContext) -> pd.DataFrame:
    ohlcv = ctx.full_ohlcv
    ix = ensure_exchange_tz_index(ohlcv.index, ctx.exchange_timezone)
    session = _session_dates(ohlcv.index, ctx.exchange_timezone)
    ts = pd.Series(ix, index=ohlcv.index)
    grouped = ts.groupby(session, sort=False)
    first = grouped.transform("min")
    last = grouped.transform("max")
    since_open = (ts - first).dt.total_seconds() / 60.0
    to_close = (last - ts).dt.total_seconds() / 60.0
    duration_min = ((last - first).dt.total_seconds() / 60.0).clip(lower=1.0)
    progress = since_open / duration_min
    sin_time = np.sin(_TWO_PI * progress.to_numpy(dtype=float))
    cos_time = np.cos(_TWO_PI * progress.to_numpy(dtype=float))
    pos = ctx.entry_positions
    return pd.DataFrame(
        {
            "minutes_since_open": _slice_at_entries(since_open, pos),
            "minutes_to_close": _slice_at_entries(to_close, pos),
            "sin_time": _slice_at_entries(
                pd.Series(sin_time, index=ohlcv.index),
                pos,
            ),
            "cos_time": _slice_at_entries(
                pd.Series(cos_time, index=ohlcv.index),
                pos,
            ),
        },
        index=pos.index,
    )


def build_session_day_of_week(ctx: EntryFeatureContext) -> pd.DataFrame:
    """Cyclical weekday encoding from entry session date (exchange TZ)."""
    ix = ensure_exchange_tz_index(pd.DatetimeIndex(ctx.labels.index), ctx.exchange_timezone)
    weekday = pd.Series(ix.dayofweek, index=ctx.labels.index, dtype=np.int64)
    if bool((weekday > 4).any()):
        bad = sorted(weekday[weekday > 4].unique().tolist())
        raise ValueError(
            "session_day_of_week: entry timestamps fall on weekend "
            f"(dayofweek {bad}; expected Mon–Fri 0–4)",
        )
    wd_frac = weekday.astype(np.float64) / 5.0
    sin_dow = np.sin(_TWO_PI * wd_frac.to_numpy(dtype=float))
    cos_dow = np.cos(_TWO_PI * wd_frac.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "sin_dow": sin_dow,
            "cos_dow": cos_dow,
        },
        index=ctx.labels.index,
    )


def build_volume_context(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    w = fc.volume_median_window_bars
    aligned = ctx.full_ohlcv
    if "volume" in aligned.columns:
        vol = aligned["volume"].astype(np.float64)
    else:
        vol = pd.Series(0.0, index=aligned.index, dtype=np.float64)
    med = vol.rolling(window=w, min_periods=w).median()
    rel = vol / med.clip(lower=1e-12)
    log_rel = pd.Series(np.log(rel.clip(lower=1e-12).to_numpy(dtype=float)), index=vol.index)
    pos = ctx.entry_positions
    return pd.DataFrame(
        {
            "rel_volume": _slice_at_entries(rel, pos),
            "log_rel_volume": _slice_at_entries(log_rel, pos),
        },
        index=pos.index,
    )


def build_vwap_distance(ctx: EntryFeatureContext) -> pd.DataFrame:
    ohlcv = ctx.full_ohlcv
    session = _session_dates(ohlcv.index, ctx.exchange_timezone)
    close = ohlcv["close"].astype(np.float64)
    hi = ohlcv["high"].astype(np.float64)
    lo = ohlcv["low"].astype(np.float64)
    if "volume" in ohlcv.columns:
        vol = ohlcv["volume"].astype(np.float64)
    else:
        vol = pd.Series(0.0, index=ohlcv.index, dtype=np.float64)
    typical = (hi + lo + close) / 3.0
    pv = typical * vol
    frame = pd.DataFrame({"session": session, "pv": pv, "vol": vol}, index=ohlcv.index)
    grouped = frame.groupby("session", sort=False)
    cum_pv = grouped["pv"].cumsum()
    cum_vol = grouped["vol"].cumsum()
    vwap = cum_pv / cum_vol.clip(lower=1e-12)
    dist = (close - vwap) / vwap.clip(lower=1e-12)
    pos = ctx.entry_positions
    return pd.DataFrame(
        {"vwap_session_dist_pct": _slice_at_entries(dist, pos)},
        index=pos.index,
    )


def g2_warmup_bars(fc: FeatureConfig) -> int:
    if fc.volume_context:
        return fc.volume_median_window_bars
    return 0
