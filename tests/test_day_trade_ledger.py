"""Day-trade ledger: rolling weekday window and cap."""

from __future__ import annotations

from datetime import date

import pytest

from sparkles.risk.day_trade_ledger import (
    DayTradeLedger,
    anchor_us_weekday_date,
    rolling_us_business_days_ending,
)


def test_anchor_saturday_rolls_to_friday() -> None:
    assert anchor_us_weekday_date(date(2024, 1, 6)) == date(2024, 1, 5)  # Sat -> Fri


def test_rolling_window_five_days_ending_monday() -> None:
    # Mon 2024-01-08; five weekdays ending Mon = Mon .. prior Tue
    w = rolling_us_business_days_ending(date(2024, 1, 8), 5)
    assert w == frozenset(
        {
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
        },
    )


def test_cap_blocks_fourth_trade_in_window() -> None:
    ledger = DayTradeLedger(max_day_trades=3, rolling_business_days=5)
    d0 = date(2024, 1, 8)
    for _ in range(3):
        assert ledger.record_if_allowed(d0) is True
    assert ledger.record_if_allowed(d0) is False
    assert ledger.count_in_window(d0) == 3


def test_old_events_drop_out_of_window() -> None:
    ledger = DayTradeLedger(max_day_trades=3, rolling_business_days=5)
    old = date(2024, 1, 2)
    ledger.record(old)
    ledger.record(old)
    ledger.record(old)
    assert ledger.count_in_window(old) == 3
    # Two weeks later, Jan 2 should be outside 5-day window ending Jan 19
    later = date(2024, 1, 19)
    assert ledger.count_in_window(later) == 0
    assert ledger.can_add_day_trade(later) is True


def test_invalid_constructor() -> None:
    with pytest.raises(ValueError, match="max_day_trades"):
        DayTradeLedger(max_day_trades=0, rolling_business_days=5)
    with pytest.raises(ValueError, match="rolling_business_days"):
        DayTradeLedger(max_day_trades=3, rolling_business_days=0)
