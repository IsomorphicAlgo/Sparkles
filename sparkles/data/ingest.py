"""Historical OHLCV ingest: chunked TwelveData pulls and Parquet cache."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sparkles.data.twelvedata_client import fetch_ohlcv
from sparkles.env import load_dotenv
from sparkles.data.symbol_hints import validate_volatility_proxy_symbol

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)

INGEST_INTERVALS = frozenset({"1min", "1day"})


def require_api_key() -> str:
    load_dotenv()
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        msg = (
            "Missing TWELVEDATA_API_KEY. Set it in the environment or a .env file "
            "(see .env.example). Never commit the real key."
        )
        raise ValueError(msg)
    return key


def symbol_parquet_path(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Path under cache_dir for one symbol/interval over [data_start, data_end]."""
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    name = (
        f"{symbol.upper()}_{interval}_{cfg.data_start.isoformat()}_"
        f"{cfg.data_end.isoformat()}.parquet"
    )
    return cache / name


def parquet_cache_path(cfg: ExperimentConfig, base_dir: Path | None = None) -> Path:
    """Main experiment symbol 1m cache path (backward compatible)."""
    return symbol_parquet_path(cfg, cfg.symbol, "1min", base_dir=base_dir)


def _context_spec_for_ticker(cfg: ExperimentConfig, ticker: str):
    want = ticker.upper()
    for spec in cfg.context_ingest.symbols:
        if spec.symbol.upper() == want:
            return spec
    return None


def resolve_ingest_target(
    cfg: ExperimentConfig,
    symbol: str | None = None,
    interval: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve CLI/API symbol, interval, and optional TwelveData exchange."""
    sym = (symbol or cfg.symbol).strip().upper()
    validate_volatility_proxy_symbol(sym)
    if interval is not None:
        iv = interval.strip()
    elif sym == cfg.symbol.upper():
        iv = "1min"
    else:
        spec = _context_spec_for_ticker(cfg, sym)
        if spec is None:
            raise ValueError(
                f"No interval for {sym!r}; pass --interval or add it under "
                "context_ingest.symbols in the experiment YAML",
            )
        iv = spec.interval
    if iv not in INGEST_INTERVALS:
        raise ValueError(
            f"Unsupported interval {iv!r}; supported: {sorted(INGEST_INTERVALS)}",
        )
    exchange: str | None = cfg.twelvedata_exchange if sym == cfg.symbol.upper() else None
    spec = _context_spec_for_ticker(cfg, sym)
    if spec is not None and spec.twelvedata_exchange is not None:
        exchange = spec.twelvedata_exchange
    return sym, iv, exchange


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


def run_symbol_ingest(
    cfg: ExperimentConfig,
    *,
    symbol: str,
    interval: str,
    exchange: str | None = None,
    api_key: str | None = None,
    force_refresh: bool = False,
    base_dir: Path | None = None,
) -> Path:
    """Download one symbol/interval for cfg.data_start..cfg.data_end and write Parquet."""
    sym, iv, ex = resolve_ingest_target(cfg, symbol, interval)
    if exchange is not None:
        ex = exchange
    key = api_key if api_key is not None else require_api_key()
    out_path = symbol_parquet_path(cfg, sym, iv, base_dir=base_dir)

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
                "Pausing %.1fs between chunks (%s %s)",
                cfg.ingest_sleep_seconds_between_chunks,
                sym,
                iv,
            )
            time.sleep(cfg.ingest_sleep_seconds_between_chunks)
        logger.info(
            "Fetching %s %s %s/%s: %s .. %s",
            sym,
            iv,
            i,
            len(windows),
            w_start,
            w_end,
        )
        chunk = fetch_ohlcv(
            cfg,
            key,
            w_start,
            w_end,
            symbol=sym,
            interval=iv,
            exchange=ex,
        )
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        combined = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        logger.warning("No rows returned for %s %s; writing empty Parquet.", sym, iv)
    else:
        combined = pd.concat(frames, axis=0)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_parquet(out_path, engine="pyarrow")
    logger.info("Wrote %s rows for %s %s to %s", len(combined), sym, iv, out_path)
    return out_path


def run_ingest(
    cfg: ExperimentConfig,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    api_key: str | None = None,
    force_refresh: bool = False,
    base_dir: Path | None = None,
) -> Path:
    """Download one symbol/interval (defaults: experiment symbol, 1min).

    Each target is cached independently. A fresh RKLB cache does not skip SPY/VIX;
    run separate ingest commands per symbol (see ``--symbol`` / ``--interval``).
    """
    sym, iv, ex = resolve_ingest_target(cfg, symbol, interval)
    return run_symbol_ingest(
        cfg,
        symbol=sym,
        interval=iv,
        exchange=ex,
        api_key=api_key,
        force_refresh=force_refresh,
        base_dir=base_dir,
    )
