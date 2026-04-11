"""Rolling US business-day window and day-trade count cap (Iteration 5).

Each **day-trade event** is one same-day round trip (buy and sell the symbol on
the same **US session calendar date**). The ledger stores one record per event
(two events on the same date count separately toward the cap).

**Window:** The last ``rolling_business_days`` **weekdays** (Mon–Fri) ending at
``as_of``; **NYSE holidays are not excluded** in v1—upgrade to a market calendar
later if you need exact exchange sessions.

All same-day round-trip checks for backtest / paper / live should call this
module only. See ``plan.md`` and ``DEVELOPER.md``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def anchor_us_weekday_date(as_of: date) -> date:
    """Last weekday on or before ``as_of`` (Sat/Sun roll back to Friday)."""
    ts = pd.Timestamp(as_of)
    while int(ts.weekday()) >= 5:
        ts = ts - pd.Timedelta(days=1)
    return ts.date()


def rolling_us_business_days_ending(as_of: date, periods: int) -> frozenset[date]:
    """``periods`` weekdays ending at the anchor weekday for ``as_of``, inclusive."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    anchor = anchor_us_weekday_date(as_of)
    idx = pd.bdate_range(end=pd.Timestamp(anchor), periods=periods)
    return frozenset(ts.date() for ts in idx)


class DayTradeLedger:
    """Track day-trade events and enforce a rolling cap."""

    def __init__(
        self,
        *,
        max_day_trades: int = 3,
        rolling_business_days: int = 5,
    ) -> None:
        if max_day_trades < 1:
            raise ValueError("max_day_trades must be >= 1")
        if rolling_business_days < 1:
            raise ValueError("rolling_business_days must be >= 1")
        self._max = max_day_trades
        self._window_len = rolling_business_days
        self._events: list[date] = []

    @property
    def max_day_trades(self) -> int:
        return self._max

    @property
    def rolling_business_days(self) -> int:
        return self._window_len

    def events(self) -> tuple[date, ...]:
        """Immutable copy of recorded session dates (one per day-trade event)."""
        return tuple(self._events)

    def record(self, session_date: date) -> None:
        """Append one day-trade event on ``session_date`` (does not check cap)."""
        self._events.append(session_date)

    def count_in_window(self, as_of: date) -> int:
        """How many recorded events fall in the rolling window ending at ``as_of``."""
        window = rolling_us_business_days_ending(as_of, self._window_len)
        return sum(1 for d in self._events if d in window)

    def can_add_day_trade(self, session_date: date) -> bool:
        """True if one more event on ``session_date`` would not exceed the cap."""
        return self.count_in_window(session_date) < self._max

    def record_if_allowed(self, session_date: date) -> bool:
        """Record one event if allowed; return whether it was recorded."""
        if not self.can_add_day_trade(session_date):
            return False
        self.record(session_date)
        return True
