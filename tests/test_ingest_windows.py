"""Unit tests for calendar chunking (no network)."""

from __future__ import annotations

from datetime import date

from sparkles.data.ingest import iter_calendar_windows


def test_single_day() -> None:
    d = date(2024, 1, 15)
    w = iter_calendar_windows(d, d, chunk_calendar_days=7)
    assert w == [(d, d)]


def test_week_chunks() -> None:
    start = date(2024, 1, 1)
    end = date(2024, 1, 20)
    w = iter_calendar_windows(start, end, chunk_calendar_days=7)
    assert w[0] == (date(2024, 1, 1), date(2024, 1, 7))
    assert w[1] == (date(2024, 1, 8), date(2024, 1, 14))
    assert w[2] == (date(2024, 1, 15), date(2024, 1, 20))


def test_empty_when_inverted() -> None:
    assert iter_calendar_windows(date(2024, 2, 1), date(2024, 1, 1), 7) == []
