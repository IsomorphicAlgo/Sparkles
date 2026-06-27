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
    """Canonical cache path: one Parquet per symbol/interval (append-friendly)."""
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{symbol.upper()}_{interval}.parquet"


def legacy_symbol_parquet_path(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Pre-incremental path keyed by experiment data_start/data_end (exact match)."""
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    name = (
        f"{symbol.upper()}_{interval}_{cfg.data_start.isoformat()}_"
        f"{cfg.data_end.isoformat()}.parquet"
    )
    return cache / name


def _legacy_symbol_parquet_candidates(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> list[Path]:
    """Any dated ingest cache ``{SYMBOL}_{interval}_{start}_{end}.parquet`` on disk."""
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    sym = symbol.upper()
    out: list[Path] = []
    for path in cache.glob(f"{sym}_{interval}_*_*.parquet"):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            date.fromisoformat(parts[-2])
            date.fromisoformat(parts[-1])
        except ValueError:
            continue
        out.append(path)
    return out


def find_legacy_symbol_parquet_path(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    """Best legacy dated cache: latest ``data_end`` in filename, then newest mtime."""
    exact = legacy_symbol_parquet_path(cfg, symbol, interval, base_dir=base_dir)
    if exact.is_file():
        return exact
    candidates = _legacy_symbol_parquet_candidates(cfg, symbol, interval, base_dir=base_dir)
    if not candidates:
        return None

    def _legacy_end_date(path: Path) -> date:
        return date.fromisoformat(path.stem.split("_")[-1])

    return max(candidates, key=lambda p: (_legacy_end_date(p), p.stat().st_mtime))


def resolve_symbol_parquet_path(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Prefer canonical cache; fall back to any legacy dated file if present."""
    canonical = symbol_parquet_path(cfg, symbol, interval, base_dir=base_dir)
    if canonical.is_file():
        return canonical
    legacy = find_legacy_symbol_parquet_path(cfg, symbol, interval, base_dir=base_dir)
    if legacy is not None:
        return legacy
    return canonical


def parquet_cache_path(cfg: ExperimentConfig, base_dir: Path | None = None) -> Path:
    """Main experiment symbol 1m cache path (canonical)."""
    return symbol_parquet_path(cfg, cfg.symbol, "1min", base_dir=base_dir)


def _index_date_bounds(df: pd.DataFrame) -> tuple[date | None, date | None]:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    ix = pd.DatetimeIndex(df.index)
    return ix.min().date(), ix.max().date()


def _cache_covers_start(data_start: date, cached_start: date) -> bool:
    """Allow a few calendar days after data_start before requiring a head backfill."""
    return cached_start <= data_start + timedelta(days=4)


def _cache_covers_end(data_end: date, cached_end: date) -> bool:
    """Allow a few calendar days before data_end (last trading day vs calendar end)."""
    return cached_end + timedelta(days=4) >= data_end


def ingest_fetch_ranges(
    data_start: date,
    data_end: date,
    cached_start: date | None,
    cached_end: date | None,
) -> list[tuple[date, date]]:
    """Calendar ranges still missing from cache relative to the experiment window."""
    if cached_start is None or cached_end is None:
        return [(data_start, data_end)]
    out: list[tuple[date, date]] = []
    if not _cache_covers_start(data_start, cached_start):
        out.append((data_start, cached_start - timedelta(days=1)))
    if not _cache_covers_end(data_end, cached_end):
        out.append((cached_end + timedelta(days=1), data_end))
    return out


def slice_ohlcv_to_experiment_range(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Trim cached OHLCV to ``data_start``..``data_end`` (exchange session dates)."""
    if df.empty:
        return df
    from sparkles.features.volatility import ensure_exchange_tz_index

    ix = ensure_exchange_tz_index(pd.DatetimeIndex(df.index), cfg.exchange_timezone)
    out = df.copy()
    out.index = ix
    tz = cfg.exchange_timezone
    start = pd.Timestamp(cfg.data_start).tz_localize(tz)
    end = pd.Timestamp(cfg.data_end).tz_localize(tz)
    session = pd.DatetimeIndex(ix).normalize()
    mask = (session >= start) & (session <= end)
    return out.loc[mask]


def load_symbol_ohlcv(
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """Load cached OHLCV for one symbol/interval, sliced to the experiment window."""
    path = resolve_symbol_parquet_path(cfg, symbol, interval, base_dir=base_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ingest Parquet not found: {path}. "
            f"Run `sparkles ingest -c …`"
            + (f" --symbol {symbol.upper()} --interval {interval}" if symbol.upper() != cfg.symbol.upper() else ""),
        )
    df = pd.read_parquet(path)
    return slice_ohlcv_to_experiment_range(df, cfg)


def load_parquet_cache(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """Load main experiment symbol 1m OHLCV sliced to ``data_start``..``data_end``."""
    return load_symbol_ohlcv(cfg, cfg.symbol, "1min", base_dir=base_dir)


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


def _read_existing_cache(
    canonical: Path,
    cfg: ExperimentConfig,
    symbol: str,
    interval: str,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    if canonical.is_file():
        return pd.read_parquet(canonical)
    legacy = find_legacy_symbol_parquet_path(cfg, symbol, interval, base_dir=base_dir)
    if legacy is not None:
        logger.info(
            "Using legacy dated cache %s → will write %s on update",
            legacy.name,
            canonical.name,
        )
        return pd.read_parquet(legacy)
    return None


def _merge_ohlcv(existing: pd.DataFrame | None, frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        parts.append(existing)
    parts.extend(f for f in frames if not f.empty)
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    combined = pd.concat(parts, axis=0)
    combined = combined.sort_index()
    return combined[~combined.index.duplicated(keep="last")]


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
    """Download/append one symbol/interval up to ``cfg.data_end`` (incremental)."""
    sym, iv, ex = resolve_ingest_target(cfg, symbol, interval)
    if exchange is not None:
        ex = exchange
    key = api_key if api_key is not None else require_api_key()
    out_path = symbol_parquet_path(cfg, sym, iv, base_dir=base_dir)

    existing = _read_existing_cache(out_path, cfg, sym, iv, base_dir=base_dir)
    cached_start, cached_end = _index_date_bounds(existing) if existing is not None else (None, None)

    if force_refresh:
        fetch_ranges = [(cfg.data_start, cfg.data_end)]
    elif cached_start is None:
        fetch_ranges = [(cfg.data_start, cfg.data_end)]
    else:
        fetch_ranges = ingest_fetch_ranges(
            cfg.data_start,
            cfg.data_end,
            cached_start,
            cached_end,
        )
        if not fetch_ranges:
            if existing is not None and not out_path.is_file():
                existing.to_parquet(out_path, engine="pyarrow")
                logger.info("Migrated legacy cache to canonical path: %s", out_path)
            return out_path if out_path.is_file() else (
                find_legacy_symbol_parquet_path(cfg, sym, iv, base_dir=base_dir) or out_path
            )

    if not fetch_ranges:
        return out_path

    frames: list[pd.DataFrame] = []
    chunk_days = cfg.ingest_chunk_calendar_days
    for r_start, r_end in fetch_ranges:
        windows = iter_calendar_windows(r_start, r_end, chunk_days)
        if not windows:
            continue
        for i, (w_start, w_end) in enumerate(windows, start=1):
            if frames and cfg.ingest_sleep_seconds_between_chunks > 0:
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

    if force_refresh and existing is not None and not existing.empty:
        # Replace overlapping experiment window; keep bars outside it.
        from sparkles.features.volatility import ensure_exchange_tz_index

        ix = ensure_exchange_tz_index(
            pd.DatetimeIndex(existing.index),
            cfg.exchange_timezone,
        )
        session = pd.DatetimeIndex(ix).normalize()
        start = pd.Timestamp(cfg.data_start).tz_localize(cfg.exchange_timezone)
        end = pd.Timestamp(cfg.data_end).tz_localize(cfg.exchange_timezone)
        keep = existing.loc[(session < start) | (session > end)]
        existing = keep if not keep.empty else None

    combined = _merge_ohlcv(existing, frames)
    if combined.empty:
        logger.warning("No rows returned for %s %s; writing empty Parquet.", sym, iv)
    combined.to_parquet(out_path, engine="pyarrow")
    c0, c1 = _index_date_bounds(combined)
    logger.info(
        "Wrote %s rows for %s %s to %s (cache span %s .. %s)",
        len(combined),
        sym,
        iv,
        out_path,
        c0,
        c1,
    )
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

    Each target is cached independently in one append-friendly Parquet file.
    Extending ``data_end`` in YAML fetches only missing calendar days.
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
