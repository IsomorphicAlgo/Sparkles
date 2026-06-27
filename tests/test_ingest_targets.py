"""Ingest target resolution and per-symbol cache paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import (
    ingest_fetch_ranges,
    legacy_symbol_parquet_path,
    load_symbol_ohlcv,
    parquet_cache_path,
    resolve_ingest_target,
    resolve_symbol_parquet_path,
    run_symbol_ingest,
    slice_ohlcv_to_experiment_range,
    symbol_parquet_path,
)


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "RKLB",
        "data_start": date(2022, 1, 1),
        "data_end": date(2026, 3, 30),
        "ingest_sleep_seconds_between_chunks": 0,
        "context_ingest": {
            "symbols": [
                {"symbol": "SPY", "interval": "1min"},
                {"symbol": "VIXY", "interval": "1day", "twelvedata_exchange": "CBOE"},
            ],
        },
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_parquet_cache_path_is_main_symbol_1min() -> None:
    cfg = _cfg()
    p = parquet_cache_path(cfg)
    assert p.name == "RKLB_1min.parquet"


def test_symbol_parquet_path_spy_vix() -> None:
    cfg = _cfg()
    spy = symbol_parquet_path(cfg, "SPY", "1min")
    vix = symbol_parquet_path(cfg, "VIXY", "1day")
    assert spy.name == "SPY_1min.parquet"
    assert vix.name == "VIXY_1day.parquet"


def test_legacy_path_fallback() -> None:
    cfg = _cfg()
    legacy = legacy_symbol_parquet_path(cfg, "SPY", "1min")
    assert legacy.name == "SPY_1min_2022-01-01_2026-03-30.parquet"


def test_ingest_fetch_ranges_tail_only() -> None:
    ranges = ingest_fetch_ranges(
        date(2022, 1, 1),
        date(2026, 5, 30),
        date(2022, 1, 3),
        date(2026, 3, 27),
    )
    assert ranges == [(date(2026, 3, 28), date(2026, 5, 30))]


def test_slice_ohlcv_to_experiment_range() -> None:
    cfg = _cfg(data_end=date(2026, 1, 5))
    tz = "America/New_York"
    idx = pd.date_range("2026-01-02 09:30", periods=780, freq="1min", tz=tz)
    df = pd.DataFrame({"close": 1.0}, index=idx)
    sliced = slice_ohlcv_to_experiment_range(df, cfg)
    assert sliced.index.max().date() <= date(2026, 1, 5)


def test_resolve_defaults_to_experiment_symbol() -> None:
    cfg = _cfg()
    sym, iv, _ex = resolve_ingest_target(cfg)
    assert sym == "RKLB"
    assert iv == "1min"


def test_resolve_infers_context_interval() -> None:
    cfg = _cfg()
    sym, iv, ex = resolve_ingest_target(cfg, symbol="VIXY")
    assert sym == "VIXY"
    assert iv == "1day"
    assert ex == "CBOE"


def test_resolve_rejects_spot_vix() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="TwelveData does not provide spot VIX"):
        resolve_ingest_target(cfg, symbol="VIX")
    with pytest.raises(ValueError, match="TwelveData does not provide spot VIX"):
        resolve_ingest_target(cfg, symbol="^VIX")


def test_resolve_requires_interval_for_unknown_symbol() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="No interval"):
        resolve_ingest_target(cfg, symbol="QQQ")


def test_run_symbol_ingest_skips_http_when_cache_covers_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    spy_path = symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    spy_path.parent.mkdir(parents=True, exist_ok=True)
    tz = "America/New_York"
    idx = pd.date_range("2022-01-03 09:30", "2026-03-27 16:00", freq="1min", tz=tz)
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx).to_parquet(
        spy_path,
    )

    fetch_calls: list[str] = []

    def _fake_fetch(*_a, **kw):
        fetch_calls.append(kw.get("symbol", ""))
        return pd.DataFrame()

    monkeypatch.setattr("sparkles.data.ingest.fetch_ohlcv", _fake_fetch)

    out = run_symbol_ingest(cfg, symbol="SPY", interval="1min", base_dir=tmp_path)
    assert out == spy_path
    assert fetch_calls == []


def test_run_symbol_ingest_fetches_tail_when_data_end_extended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(data_end=date(2026, 5, 30))
    spy_path = symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    spy_path.parent.mkdir(parents=True, exist_ok=True)
    tz = "America/New_York"
    idx = pd.date_range("2022-01-03 09:30", "2026-03-27 16:00", freq="1min", tz=tz)
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx).to_parquet(
        spy_path,
    )

    fetched: list[tuple[date, date]] = []

    def _fake_fetch(_cfg, _key, w_start, w_end, **kw):
        fetched.append((w_start, w_end))
        ts = pd.Timestamp(w_start, tz=tz) + pd.Timedelta(hours=10)
        return pd.DataFrame(
            {"open": [2.0], "high": [2.0], "low": [2.0], "close": [2.0], "volume": [2.0]},
            index=[ts],
        )

    monkeypatch.setattr("sparkles.data.ingest.fetch_ohlcv", _fake_fetch)

    run_symbol_ingest(cfg, symbol="SPY", interval="1min", base_dir=tmp_path)
    assert fetched
    assert any(s == date(2026, 3, 28) for s, _e in fetched)
    assert fetched[-1][1] == date(2026, 5, 30)


def test_load_symbol_ohlcv_slices_to_yaml_window(tmp_path: Path) -> None:
    cfg = _cfg(data_end=date(2026, 1, 10))
    path = symbol_parquet_path(cfg, "RKLB", "1min", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tz = "America/New_York"
    idx = pd.date_range("2026-01-02 09:30", "2026-01-20 16:00", freq="1min", tz=tz)
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx).to_parquet(
        path,
    )
    df = load_symbol_ohlcv(cfg, "RKLB", "1min", base_dir=tmp_path)
    assert df.index.max().date() <= date(2026, 1, 10)


def test_resolve_legacy_ingest_when_data_end_extended(tmp_path: Path) -> None:
    cfg = _cfg(data_end=date(2026, 5, 30))
    legacy = tmp_path / "data" / "cache" / "SPY_1min_2022-01-01_2026-03-30.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"x")
    resolved = resolve_symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    assert resolved == legacy


def test_resolve_symbol_parquet_path_prefers_canonical(tmp_path: Path) -> None:
    cfg = _cfg()
    canonical = symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    legacy = legacy_symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"x")
    legacy.write_bytes(b"y")
    assert resolve_symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path) == canonical
