"""Ingest target resolution and per-symbol cache paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import (
    parquet_cache_path,
    resolve_ingest_target,
    run_symbol_ingest,
    symbol_parquet_path,
)


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "RKLB",
        "data_start": date(2022, 1, 1),
        "data_end": date(2026, 3, 30),
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
    assert p.name == "RKLB_1min_2022-01-01_2026-03-30.parquet"


def test_symbol_parquet_path_spy_vix() -> None:
    cfg = _cfg()
    spy = symbol_parquet_path(cfg, "SPY", "1min")
    vix = symbol_parquet_path(cfg, "VIXY", "1day")
    assert spy.name == "SPY_1min_2022-01-01_2026-03-30.parquet"
    assert vix.name == "VIXY_1day_2022-01-01_2026-03-30.parquet"


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


def test_run_symbol_ingest_skips_http_when_spy_cache_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    spy_path = symbol_parquet_path(cfg, "SPY", "1min", base_dir=tmp_path)
    spy_path.parent.mkdir(parents=True, exist_ok=True)
    spy_path.write_bytes(b"cached")

    fetch_calls: list[str] = []

    def _fake_fetch(*_a, **kw):
        fetch_calls.append(kw.get("symbol", ""))
        return pd.DataFrame()

    monkeypatch.setattr("sparkles.data.ingest.fetch_ohlcv", _fake_fetch)

    out = run_symbol_ingest(cfg, symbol="SPY", interval="1min", base_dir=tmp_path)
    assert out == spy_path
    assert fetch_calls == []
