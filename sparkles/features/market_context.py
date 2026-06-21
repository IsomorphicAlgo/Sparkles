"""SPY trailing return and VIX daily change (ML expansion Phase G3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.time import entry_session_dates
from sparkles.features.volatility import ensure_exchange_tz_index


def _spy_return_column_name(bars: int) -> str:
    return f"spy_ret_{bars}m"


def build_market_context(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    if ctx.market_spy_1m is None or ctx.market_vix_1d is None:
        raise ValueError(
            "market_context requires market_spy_1m and market_vix_1d on EntryFeatureContext",
        )
    bars = fc.market_spy_return_bars
    col_spy = _spy_return_column_name(bars)

    spy = ctx.market_spy_1m
    spy_ix = ensure_exchange_tz_index(spy.index, ctx.exchange_timezone)
    spy_close = spy["close"].astype(np.float64)
    spy_close.index = spy_ix
    entry_ix = ensure_exchange_tz_index(ctx.labels.index, ctx.exchange_timezone)
    spy_pos = pd.Series(
        spy_close.index.get_indexer(entry_ix),
        index=ctx.labels.index,
        dtype=np.int64,
    )
    pos_arr = spy_pos.to_numpy(dtype=np.int64, copy=False)
    close_arr = spy_close.to_numpy(dtype=np.float64, copy=False)
    ret = np.full(len(pos_arr), np.nan, dtype=np.float64)
    ok = pos_arr >= bars
    if bool(ok.any()):
        cur = close_arr[pos_arr[ok]]
        lag = close_arr[pos_arr[ok] - bars]
        ret[ok] = np.log(cur / np.clip(lag, 1e-12, None))
    spy_ret = pd.Series(ret, index=ctx.labels.index)

    vix = ctx.market_vix_1d
    vix_ix = ensure_exchange_tz_index(vix.index, ctx.exchange_timezone)
    daily_close = (
        vix["close"]
        .astype(np.float64)
        .groupby(vix_ix.normalize())
        .last()
        .sort_index()
    )
    daily_chg = daily_close.pct_change(1).shift(1)
    chg_by_date = pd.Series(
        daily_chg.to_numpy(dtype=float),
        index=pd.DatetimeIndex(daily_chg.index).date,
    )
    session = entry_session_dates(ctx.labels.index, ctx.exchange_timezone)
    vix_chg = session.map(chg_by_date)

    return pd.DataFrame(
        {
            col_spy: spy_ret,
            "vix_chg_1d": pd.Series(
                vix_chg.to_numpy(dtype=float),
                index=ctx.labels.index,
            ),
        },
        index=ctx.labels.index,
    )


def g3_warmup_bars(fc: FeatureConfig) -> int:
    if fc.market_context:
        return fc.market_spy_return_bars
    return 0
