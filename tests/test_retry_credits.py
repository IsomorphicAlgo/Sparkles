"""TwelveData credit-error classification."""

from __future__ import annotations

from twelvedata.exceptions import BadRequestError, TwelveDataError

from sparkles.data.retry import is_no_data_in_range_error, is_per_minute_credit_exhausted_error


def test_detects_user_message() -> None:
    msg = (
        "You have run out of API credits for the current minute. "
        "13 API credits were used, with the current limit being 8."
    )
    assert is_per_minute_credit_exhausted_error(TwelveDataError(msg)) is True


def test_ignores_unrelated() -> None:
    exc = TwelveDataError("Invalid symbol")
    assert is_per_minute_credit_exhausted_error(exc) is False


def test_no_data_in_range_weekend() -> None:
    exc = BadRequestError(
        '{"code":400,"message":"No data is available on the specified dates."}',
    )
    assert is_no_data_in_range_error(exc) is True
