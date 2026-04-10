"""Trading-day realized volatility from daily closes, aligned to 1m bars (no lookahead).

Each 1m bar on session date *D* shares one estimate: rolling std of **daily** log
returns through the prior session's close (``rolling(...).std().shift(1)``).

If you change ``vol_lookback_trading_days``, update YAML and triple_barrier code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig

TRADING_DAYS_PER_YEAR: int = 252


def ensure_exchange_tz_index(
    index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> pd.DatetimeIndex:
    """Interpret naive index as *exchange_timezone* wall time; else convert."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if index.tz is None:
        return index.tz_localize(
            exchange_timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return index.tz_convert(exchange_timezone)


def daily_last_close(
    ohlcv: pd.DataFrame,
    *,
    exchange_timezone: str,
    close_col: str = "close",
) -> pd.Series:
    """Last 1m close per session date in *exchange_timezone*."""
    if close_col not in ohlcv.columns:
        raise KeyError(f"Expected column {close_col!r} in OHLCV frame")
    ix = ensure_exchange_tz_index(ohlcv.index, exchange_timezone)
    s = ohlcv[close_col].copy()
    s.index = ix
    session_open = s.index.normalize()
    return s.groupby(session_open, sort=True).last()


def daily_log_returns(daily_close: pd.Series) -> pd.Series:
    """Log returns on the daily close series (aligned to daily index)."""
    c = daily_close.sort_index()
    ratio = c / c.shift(1)
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=c.index, dtype=float)


def rolling_volatility_daily_returns_no_lookahead(
    daily_close: pd.Series,
    *,
    lookback_trading_days: int,
    annualize: bool = True,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.Series, pd.Series]:
    """Rolling std of daily log returns, then shift(1) (no same-day close leak).

    Returns (sigma_daily, sigma_ann or duplicate of daily if not annualize).
    """
    if lookback_trading_days < 2:
        raise ValueError("lookback_trading_days must be at least 2")
    r = daily_log_returns(daily_close)
    sig_d = r.rolling(
        window=lookback_trading_days,
        min_periods=lookback_trading_days,
    ).std()
    sig_d = sig_d.shift(1)
    if annualize:
        sig_a = sig_d * np.sqrt(float(trading_days_per_year))
    else:
        sig_a = sig_d
    return sig_d, sig_a


def align_volatility_to_1m_index(
    ohlcv: pd.DataFrame,
    sigma_ann_by_session: pd.Series,
    *,
    exchange_timezone: str,
) -> pd.Series:
    """Map daily sigma to each 1m bar by session date (exchange TZ)."""
    ix = ensure_exchange_tz_index(ohlcv.index, exchange_timezone)
    session = ix.normalize()
    # sigma_ann_by_session is indexed by normalized session timestamps; align labels
    aligned = sigma_ann_by_session.reindex(session)
    aligned.index = ohlcv.index
    return aligned


def add_volatility_columns(
    ohlcv: pd.DataFrame,
    *,
    exchange_timezone: str,
    lookback_trading_days: int,
    annualize: bool = True,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    close_col: str = "close",
) -> pd.DataFrame:
    """Copy of *ohlcv* with ``sigma_daily_{N}d`` and ``vol_{N}d_ann`` columns."""
    daily = daily_last_close(
        ohlcv,
        exchange_timezone=exchange_timezone,
        close_col=close_col,
    )
    sig_d, sig_a = rolling_volatility_daily_returns_no_lookahead(
        daily,
        lookback_trading_days=lookback_trading_days,
        annualize=annualize,
        trading_days_per_year=trading_days_per_year,
    )
    col_d = f"sigma_daily_{lookback_trading_days}d"
    col_a = f"vol_{lookback_trading_days}d_ann" if annualize else col_d

    out: pd.DataFrame = ohlcv.copy()
    out[col_d] = align_volatility_to_1m_index(
        ohlcv,
        sig_d,
        exchange_timezone=exchange_timezone,
    )
    out[col_a] = align_volatility_to_1m_index(
        ohlcv,
        sig_a,
        exchange_timezone=exchange_timezone,
    )
    return out


def add_volatility_from_config(
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    close_col: str = "close",
) -> pd.DataFrame:
    """Use ``cfg.exchange_timezone`` and ``cfg.vol_lookback_trading_days``."""
    return add_volatility_columns(
        ohlcv,
        exchange_timezone=cfg.exchange_timezone,
        lookback_trading_days=cfg.vol_lookback_trading_days,
        annualize=True,
        close_col=close_col,
    )
