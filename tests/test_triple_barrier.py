"""Triple-barrier labeling: tie-break, vertical by trading day, end-of-data."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.labels.triple_barrier import (
    build_triple_barrier_labels,
    labeled_parquet_path,
    run_label,
)


def _cfg(**overrides: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "TEST",
        "data_start": date(2024, 1, 1),
        "data_end": date(2024, 12, 31),
        "profit_barrier_base": 0.10,
        "stop_loss_base": 0.05,
        "min_profit_per_trade_pct": 0.01,
        "vol_lookback_trading_days": 20,
        "vertical_max_trading_days": 10,
        "label_entry_stride": 1,
        "barrier_vol_scale_min": 0.25,
        "barrier_vol_scale_max": 4.0,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def test_same_bar_stop_before_take_profit() -> None:
    """Pessimistic long: if both SL and TP touched on one bar, SL wins."""
    tz = "America/New_York"
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min", tz=tz)
    df = pd.DataFrame(
        {
            "high": [100.0, 120.0, 100.0],
            "low": [100.0, 94.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vol_20d_ann": [0.5, 0.5, 0.5],
        },
        index=idx,
    )
    out = build_triple_barrier_labels(df, _cfg())
    assert len(out) == 2
    row0 = out.iloc[0]
    assert row0["barrier_outcome"] == "stop_loss"
    assert int(row0["bars_forward"]) == 1


def test_take_profit_when_high_crosses_only() -> None:
    tz = "America/New_York"
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min", tz=tz)
    df = pd.DataFrame(
        {
            "high": [100.0, 112.0, 100.0],
            "low": [100.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vol_20d_ann": [0.5, 0.5, 0.5],
        },
        index=idx,
    )
    out = build_triple_barrier_labels(df, _cfg())
    assert out.iloc[0]["barrier_outcome"] == "take_profit"
    assert int(out.iloc[0]["bars_forward"]) == 1


def test_vertical_fires_before_ohlc_on_next_trading_day() -> None:
    """With vertical_max=1, first bar of day 2 exits before that bar's high/low."""
    tz = "America/New_York"
    d1 = pd.date_range("2024-01-02 09:30", periods=2, freq="1min", tz=tz)
    d2 = pd.date_range("2024-01-03 09:30", periods=1, freq="1min", tz=tz)
    idx = pd.DatetimeIndex(d1.tolist() + d2.tolist())
    df = pd.DataFrame(
        {
            "high": [100.0, 100.0, 200.0],
            "low": [100.0, 100.0, 50.0],
            "close": [100.0, 100.0, 100.0],
            "vol_20d_ann": [0.5, 0.5, 0.5],
        },
        index=idx,
    )
    out = build_triple_barrier_labels(df, _cfg(vertical_max_trading_days=1))
    row = out.iloc[0]
    assert row["barrier_outcome"] == "vertical"
    assert int(row["bars_forward"]) == 2


def test_end_of_data_when_no_barrier_before_series_end() -> None:
    tz = "America/New_York"
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min", tz=tz)
    df = pd.DataFrame(
        {
            "high": [100.0, 100.5, 100.1],
            "low": [100.0, 99.5, 99.9],
            "close": [100.0, 100.2, 100.0],
            "vol_20d_ann": [0.5, 0.5, 0.5],
        },
        index=idx,
    )
    out = build_triple_barrier_labels(df, _cfg(vertical_max_trading_days=50))
    row = out.iloc[0]
    assert row["barrier_outcome"] == "end_of_data"
    assert int(row["bars_forward"]) == 2


def test_labeled_parquet_path_includes_stride(tmp_path) -> None:
    cfg = _cfg(label_entry_stride=390)
    p = labeled_parquet_path(cfg, base_dir=tmp_path)
    assert "_s390.parquet" in p.name


def test_run_label_roundtrip(tmp_path) -> None:
    tz = "America/New_York"
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min", tz=tz)
    df = pd.DataFrame(
        {
            "high": [100.0, 112.0, 100.0],
            "low": [100.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vol_20d_ann": [0.5, 0.5, 0.5],
        },
        index=idx,
    )
    cfg = _cfg(
        symbol="XY",
        data_start=date(2024, 1, 2),
        data_end=date(2024, 1, 3),
    )
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    pq = parquet_cache_path(cfg, base_dir=tmp_path)
    assert pq.parent == cache
    df.to_parquet(pq)
    out = run_label(cfg, base_dir=tmp_path)
    assert out.is_file()
    lab = pd.read_parquet(out)
    assert lab["barrier_outcome"].iloc[0] == "take_profit"
