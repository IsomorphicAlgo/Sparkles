"""Session calendar helpers shared across feature builders."""

from __future__ import annotations

import pandas as pd

from sparkles.features.volatility import ensure_exchange_tz_index


def entry_session_dates(
    index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> pd.Series:
    """US session calendar date per bar (normalized midnight in exchange TZ)."""
    ix = ensure_exchange_tz_index(pd.DatetimeIndex(index), exchange_timezone)
    norm = pd.DatetimeIndex(ix).normalize()
    return pd.Series(norm.date, index=index, dtype=object)
