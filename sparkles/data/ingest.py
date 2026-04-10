"""Historical 1m OHLCV ingest: chunked TwelveData pulls and Parquet cache."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sparkles.data.twelvedata_client import fetch_ohlcv_1min

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


def require_api_key() -> str:
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        msg = (
            "Missing TWELVEDATA_API_KEY. Set it in the environment or a .env file "
            "(see .env.example). Never commit the real key."
        )
        raise ValueError(msg)
    return key


def parquet_cache_path(cfg: ExperimentConfig, base_dir: Path | None = None) -> Path:
    """Path under cache_dir for this symbol and [data_start, data_end] range."""
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    name = (
        f"{cfg.symbol.upper()}_1min_{cfg.data_start.isoformat()}_"
        f"{cfg.data_end.isoformat()}.parquet"
    )
    return cache / name


def iter_calendar_windows(
    start: date,
    end: date,
    chunk_calendar_days: int,
) -> list[tuple[date, date]]:
    """Non-overlapping inclusive calendar windows covering [start, end]."""
    if end < start:
        return []
    out: list[tuple[date, date]] = []
    cur = start
    span = timedelta(days=chunk_calendar_days)
    one = timedelta(days=1)
    while cur <= end:
        chunk_end = min(end, cur + span - one)
        out.append((cur, chunk_end))
        cur = chunk_end + one
    return out


def run_ingest(
    cfg: ExperimentConfig,
    *,
    api_key: str | None = None,
    force_refresh: bool = False,
    base_dir: Path | None = None,
) -> Path:
    """Download 1m bars for cfg.data_start..cfg.data_end and write Parquet.

    Returns path to the Parquet file. Skips HTTP when cache exists and is newer
    than cache_ttl_hours unless force_refresh is True.
    """
    key = api_key if api_key is not None else require_api_key()
    out_path = parquet_cache_path(cfg, base_dir=base_dir)

    if out_path.is_file() and not force_refresh:
        age_h = (time.time() - out_path.stat().st_mtime) / 3600.0
        if age_h < cfg.cache_ttl_hours:
            logger.info(
                "Using cached Parquet (%.1f h old, TTL %s h): %s",
                age_h,
                cfg.cache_ttl_hours,
                out_path,
            )
            return out_path

    windows = iter_calendar_windows(
        cfg.data_start,
        cfg.data_end,
        cfg.ingest_chunk_calendar_days,
    )
    if not windows:
        raise ValueError("data_start/data_end produced no ingest windows")

    frames: list[pd.DataFrame] = []
    for i, (w_start, w_end) in enumerate(windows, start=1):
        if i > 1 and cfg.ingest_sleep_seconds_between_chunks > 0:
            logger.info(
                "Pausing %.1fs between chunks (TwelveData rate / credit limits)",
                cfg.ingest_sleep_seconds_between_chunks,
            )
            time.sleep(cfg.ingest_sleep_seconds_between_chunks)
        logger.info(
            "Fetching %s/%s: %s .. %s",
            i,
            len(windows),
            w_start,
            w_end,
        )
        chunk = fetch_ohlcv_1min(cfg, key, w_start, w_end)
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        combined = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        logger.warning("No rows returned for any window; writing empty Parquet.")
    else:
        combined = pd.concat(frames, axis=0)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_parquet(out_path, engine="pyarrow")
    logger.info("Wrote %s rows to %s", len(combined), out_path)
    return out_path
