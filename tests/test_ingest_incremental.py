"""Unit tests for incremental ingest gap logic (no network)."""

from __future__ import annotations

from datetime import date

from sparkles.data.ingest import ingest_fetch_ranges


def test_full_range_when_empty_cache() -> None:
    assert ingest_fetch_ranges(
        date(2022, 1, 1),
        date(2026, 5, 30),
        None,
        None,
    ) == [(date(2022, 1, 1), date(2026, 5, 30))]


def test_no_gaps_when_cache_covers() -> None:
    assert ingest_fetch_ranges(
        date(2022, 1, 1),
        date(2026, 3, 30),
        date(2022, 1, 1),
        date(2026, 3, 30),
    ) == []


def test_head_and_tail_gaps() -> None:
    assert ingest_fetch_ranges(
        date(2022, 1, 1),
        date(2026, 5, 30),
        date(2022, 6, 1),
        date(2026, 3, 30),
    ) == [
        (date(2022, 1, 1), date(2022, 5, 31)),
        (date(2026, 3, 31), date(2026, 5, 30)),
    ]
