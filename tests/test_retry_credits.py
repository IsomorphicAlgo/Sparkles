"""TwelveData credit-error classification."""

from __future__ import annotations

from twelvedata.exceptions import TwelveDataError

from sparkles.data.retry import is_per_minute_credit_exhausted_error


def test_detects_user_message() -> None:
    msg = (
        "You have run out of API credits for the current minute. "
        "13 API credits were used, with the current limit being 8."
    )
    assert is_per_minute_credit_exhausted_error(TwelveDataError(msg)) is True


def test_ignores_unrelated() -> None:
    exc = TwelveDataError("Invalid symbol")
    assert is_per_minute_credit_exhausted_error(exc) is False
