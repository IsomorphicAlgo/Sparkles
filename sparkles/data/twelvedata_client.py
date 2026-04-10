"""TwelveData SDK wrapper with resilient HTTP and normalized OHLCV frames."""

from __future__ import annotations

import logging
from datetime import date
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import requests
from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError
from twelvedata.http_client import DefaultHttpClient

from sparkles.data.retry import (
    RetryPolicy,
    is_retryable_requests_error,
    is_retryable_twelvedata_error,
    sleep_after_twelvedata_retry,
)

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


class ResilientHttpClient(DefaultHttpClient):  # type: ignore[misc]
    """HTTP client with configurable timeout, retries, and Retry-After support."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        per_minute_credit_wait_seconds: float = 65.0,
    ) -> None:
        super().__init__(base_url)
        self._timeout = timeout
        self._retry_policy = retry_policy or RetryPolicy()
        self._per_minute_credit_wait = per_minute_credit_wait_seconds

    def get(
        self,
        relative_url: str,
        *_unused: object,
        **kwargs: object,
    ) -> requests.Response:
        last_exc: BaseException | None = None
        for attempt in range(self._retry_policy.max_attempts):
            try:
                return self._single_get(relative_url, **kwargs)
            except (TwelveDataError, requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt + 1 >= self._retry_policy.max_attempts:
                    break
                if isinstance(e, TwelveDataError):
                    if not is_retryable_twelvedata_error(e):
                        raise
                if not (
                    is_retryable_requests_error(e) or is_retryable_twelvedata_error(e)
                ):
                    raise
                logger.warning(
                    "TwelveData request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self._retry_policy.max_attempts,
                    e,
                )
                sleep_after_twelvedata_retry(
                    e,
                    attempt,
                    self._retry_policy,
                    self._per_minute_credit_wait,
                )
        assert last_exc is not None
        raise last_exc

    def _single_get(
        self,
        relative_url: str,
        **kwargs: object,
    ) -> requests.Response:
        raw_params = kwargs.get("params", {})
        if not isinstance(raw_params, dict):
            raw_params = {}
        params: dict[str, object] = dict(raw_params)
        params["source"] = "python"

        url = f"{self.base_url}{relative_url}"
        resp = requests.get(
            url,
            timeout=self._timeout,
            params=cast(Any, params),
        )
        if (
            "Is_batch" in resp.headers and resp.headers["Is_batch"] == "true"
        ) or (
            "Content-Type" in resp.headers
            and resp.headers["Content-Type"] == "text/csv"
        ):
            return resp

        if not resp.ok:
            self._raise_from_http_response(resp)

        try:
            json_resp = resp.json()
        except JSONDecodeError:
            return resp

        if "status" not in json_resp:
            return resp

        if json_resp["status"] != "error":
            return resp

        error_code = json_resp["code"]
        try:
            message = json_resp["message"]
        except (KeyError, ValueError):
            message = resp.text

        DefaultHttpClient._raise_error(error_code, message)
        raise AssertionError("twelvedata _raise_error should have raised")

    @staticmethod
    def _raise_from_http_response(resp: requests.Response) -> None:
        """Non-2xx HTTP: map to TwelveData errors (SDK only maps JSON body codes)."""
        code = resp.status_code
        text = resp.text
        DefaultHttpClient._raise_error(code, text)


def make_td_client(
    api_key: str,
    *,
    timeout: float,
    retry_policy: RetryPolicy,
    per_minute_credit_wait_seconds: float = 65.0,
) -> TDClient:
    return TDClient(
        api_key,
        http_client=ResilientHttpClient(
            "https://api.twelvedata.com",
            timeout=timeout,
            retry_policy=retry_policy,
            per_minute_credit_wait_seconds=per_minute_credit_wait_seconds,
        ),
    )


def fetch_ohlcv_1min(
    cfg: ExperimentConfig,
    api_key: str,
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """Fetch 1m OHLCV for [window_start, window_end] (inclusive calendar dates)."""
    policy = RetryPolicy(max_attempts=cfg.retry_max_attempts)
    client = make_td_client(
        api_key,
        timeout=cfg.http_timeout_seconds,
        retry_policy=policy,
        per_minute_credit_wait_seconds=cfg.twelvedata_per_minute_credit_wait_seconds,
    )
    kwargs: dict[str, object] = {
        "symbol": cfg.symbol,
        "interval": "1min",
        "start_date": window_start.isoformat(),
        "end_date": window_end.isoformat(),
        "outputsize": cfg.twelvedata_outputsize,
        "order": "asc",
        "timezone": cfg.exchange_timezone,
    }
    if cfg.twelvedata_exchange is not None:
        kwargs["exchange"] = cfg.twelvedata_exchange
    ts = client.time_series(**kwargs)
    df = ts.as_pandas()
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = normalize_ohlcv_frame(df)
    return df


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex, numeric OHLCV, sorted ascending."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
