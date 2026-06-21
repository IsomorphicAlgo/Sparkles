"""Per-trade return proxies from triple-barrier labels (Phase I1)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sparkles.features.time import entry_session_dates
from sparkles.features.volatility import ensure_exchange_tz_index


def realized_return_fraction(
    outcome: str,
    *,
    tp_move_effective: float,
    sl_move: float,
    entry_close: float,
    exit_close: float | None = None,
) -> float:
    """Signed fractional return for one long entry given realized barrier outcome."""
    if outcome == "take_profit":
        return float(tp_move_effective)
    if outcome == "stop_loss":
        return -float(sl_move)
    if outcome in ("vertical", "end_of_data"):
        if exit_close is None:
            raise ValueError(f"exit_close required for outcome {outcome!r}")
        if entry_close <= 0:
            raise ValueError("entry_close must be positive")
        return float(exit_close / entry_close - 1.0)
    raise ValueError(f"Unknown barrier outcome: {outcome!r}")


def lookup_ohlcv_position(
    ohlcv_index: pd.DatetimeIndex,
    entry_time: pd.Timestamp,
) -> int | None:
    """Return integer position of ``entry_time`` in ``ohlcv_index``, or None."""
    ts = pd.Timestamp(entry_time)
    try:
        return int(ohlcv_index.get_loc(ts))
    except KeyError:
        pass
    idx = ohlcv_index.get_indexer([ts], method=None)
    if idx.size == 0 or idx[0] < 0:
        return None
    return int(idx[0])


def exit_close_at_bars_forward(
    ohlcv: pd.DataFrame,
    entry_time: pd.Timestamp,
    bars_forward: int,
    exchange_timezone: str,
) -> tuple[float | None, date | None]:
    """Close at ``bars_forward`` bars after entry and that bar's session date."""
    if bars_forward < 1:
        return None, None
    ix = ensure_exchange_tz_index(ohlcv.index, exchange_timezone)
    pos = lookup_ohlcv_position(ix, entry_time)
    if pos is None:
        return None, None
    exit_pos = pos + int(bars_forward)
    if exit_pos >= len(ix):
        return None, None
    exit_close = float(ohlcv["close"].iloc[exit_pos])
    exit_session = entry_session_dates(
        pd.DatetimeIndex([ix[exit_pos]]),
        exchange_timezone,
    ).iloc[0]
    return exit_close, exit_session


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Peak-to-trough drop on a cumulative return series (negative or zero)."""
    if cumulative_returns.empty:
        return 0.0
    peak = cumulative_returns.cummax()
    drawdown = cumulative_returns - peak
    return float(drawdown.min())
