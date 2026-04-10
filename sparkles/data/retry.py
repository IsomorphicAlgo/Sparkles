"""Backoff and classification for TwelveData / HTTP retries."""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass

import requests
from twelvedata.exceptions import (
    BadRequestError,
    InternalServerError,
    InvalidApiKeyError,
    TwelveDataError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter (full jitter cap)."""

    max_attempts: int = 6
    base_seconds: float = 1.0
    max_seconds: float = 90.0


def backoff_sleep_seconds(attempt_index: int, policy: RetryPolicy) -> float:
    """Seconds to sleep before attempt attempt_index (0-based after first failure)."""
    cap = min(policy.max_seconds, policy.base_seconds * (2**attempt_index))
    return random.uniform(0, cap)


def parse_retry_after_seconds(header_value: str | None) -> float | None:
    if not header_value:
        return None
    header_value = header_value.strip()
    if re.fullmatch(r"\d+", header_value):
        return float(header_value)
    return None


def is_retryable_twelvedata_error(exc: BaseException) -> bool:
    if isinstance(exc, InvalidApiKeyError):
        return False
    if isinstance(exc, BadRequestError):
        return False
    if isinstance(exc, InternalServerError):
        return True
    if isinstance(exc, TwelveDataError):
        msg = str(exc).lower()
        if "http 429" in msg:
            return True
        if "rate" in msg or "limit" in msg or "too many" in msg:
            return True
        if "timeout" in msg or "timed out" in msg:
            return True
        return False
    return False


def is_retryable_requests_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.Timeout):
        return True
    if isinstance(exc, requests.ConnectionError):
        return True
    return False


def is_per_minute_credit_exhausted_error(exc: BaseException) -> bool:
    """TwelveData free/basic: '8 credits per minute' style errors."""
    if not isinstance(exc, TwelveDataError):
        return False
    msg = str(exc).lower()
    if "run out of api credits" in msg and "minute" in msg:
        return True
    if "api credits" in msg and "current minute" in msg:
        return True
    if "credit" in msg and "current minute" in msg:
        return True
    return False


def sleep_before_retry(
    attempt_index: int,
    policy: RetryPolicy,
    retry_after_header: str | None = None,
) -> None:
    ra = parse_retry_after_seconds(retry_after_header)
    if ra is not None:
        time.sleep(min(ra, policy.max_seconds))
        return
    time.sleep(backoff_sleep_seconds(attempt_index, policy))


def sleep_after_twelvedata_retry(
    exc: BaseException,
    attempt_index: int,
    policy: RetryPolicy,
    per_minute_credit_wait_seconds: float,
) -> None:
    """Prefer a full-minute wait on credit exhaustion; else normal backoff."""
    if is_per_minute_credit_exhausted_error(exc):
        wait = per_minute_credit_wait_seconds + random.uniform(0, 5.0)
        logger.info(
            "TwelveData per-minute credit window hit; sleeping %.1fs before retry",
            wait,
        )
        time.sleep(wait)
        return
    sleep_before_retry(attempt_index, policy, None)
