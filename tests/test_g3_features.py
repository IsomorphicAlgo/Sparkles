"""Phase G3 microstructure and market context feature tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparkles.config.schema import (
    ContextSymbolConfig,
    ExperimentConfig,
    FeatureConfig,
)
from sparkles.data.context_ingest import context_parquet_path
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.dataset import build_feature_matrix, feature_warmup_bars
from sparkles.features.market_context import build_market_context
from sparkles.features.microstructure import build_bar_microstructure
from tests.test_dataset import _cfg


def _synthetic_ohlcv() -> pd.DataFrame:
    tz = "America/New_York"
    idx = pd.date_range("2024-06-03 09:30", periods=120, freq="1min", tz=tz)
    close = pd.Series(100.0 + np.arange(120, dtype=float) * 0.01, index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000 + np.arange(120) * 10,
        },
        index=idx,
    )


def test_bar_microstructure_close_loc_and_body() -> None:
    ohlcv = _synthetic_ohlcv()
    row = ohlcv.iloc[10]
    labels = pd.DataFrame(
        {
            "entry_close": [row["close"]],
            "barrier_outcome": ["vertical"],
            "sigma_ann_at_entry": [0.5],
            "vol_scale_ratio": [1.0],
            "tp_move_effective": [0.1],
            "sl_move": [0.05],
        },
        index=ohlcv.index[[10]],
    )
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        bar_microstructure=True,
    )
    ctx = EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=ohlcv.reindex(labels.index),
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=pd.Series([10], index=labels.index),
        feature_config=fc,
        exchange_timezone="America/New_York",
    )
    X = build_bar_microstructure(ctx)
    span = row["high"] - row["low"]
    assert X["close_loc_value"].iloc[0] == pytest.approx(
        (row["close"] - row["low"]) / span,
    )
    assert X["bar_body_pct"].iloc[0] == pytest.approx(
        (row["close"] - row["open"]) / row["close"],
    )


def _write_context_parquets(tmp_path: Path, ohlcv: pd.DataFrame) -> None:
    cfg = ExperimentConfig(
        symbol="RKLB",
        data_start=date(2024, 6, 1),
        data_end=date(2024, 6, 10),
        train_start=date(2024, 6, 3),
        train_end=date(2024, 6, 4),
        val_start=date(2024, 6, 5),
        val_end=date(2024, 6, 7),
        context_ingest={
            "symbols": [
                {"symbol": "SPY", "interval": "1min"},
                {"symbol": "VIXY", "interval": "1day", "twelvedata_exchange": "CBOE"},
            ],
        },
    )
    spy_spec = ContextSymbolConfig(symbol="SPY", interval="1min")
    vix_spec = ContextSymbolConfig(
        symbol="VIXY",
        interval="1day",
        twelvedata_exchange="CBOE",
    )
    spy_path = context_parquet_path(cfg, spy_spec, base_dir=tmp_path)
    vix_path = context_parquet_path(cfg, vix_spec, base_dir=tmp_path)
    spy_path.parent.mkdir(parents=True, exist_ok=True)
    spy = ohlcv.copy()
    spy["close"] = ohlcv["close"] * 0.5
    spy.to_parquet(spy_path)
    vix_idx = pd.date_range("2024-06-01", periods=10, freq="D", tz="America/New_York")
    vix = pd.DataFrame(
        {"open": 15.0, "high": 16.0, "low": 14.0, "close": 15.0 + np.arange(10) * 0.1, "volume": 0.0},
        index=vix_idx,
    )
    vix.to_parquet(vix_path)


def test_market_context_spy_ret_and_vix_chg(tmp_path: Path) -> None:
    ohlcv = _synthetic_ohlcv()
    _write_context_parquets(tmp_path, ohlcv)
    labels = pd.DataFrame(
        {
            "entry_close": ohlcv["close"].iloc[80:],
            "barrier_outcome": ["vertical"] * 40,
            "sigma_ann_at_entry": [0.5] * 40,
            "vol_scale_ratio": [1.0] * 40,
            "tp_move_effective": [0.1] * 40,
            "sl_move": [0.05] * 40,
        },
        index=ohlcv.index[80:],
    )
    fc = FeatureConfig(
        log_entry_close=False,
        label_geometry=False,
        intraday_range_pct=False,
        log1p_volume=False,
        market_context=True,
        market_spy_return_bars=15,
    )
    cfg = _cfg(
        features=fc,
        context_ingest={
            "symbols": [
                {"symbol": "SPY", "interval": "1min"},
                {"symbol": "VIXY", "interval": "1day", "twelvedata_exchange": "CBOE"},
            ],
        },
    )
    X, _y = build_feature_matrix(labels, ohlcv, cfg, base_dir=tmp_path)
    assert "spy_ret_15m" in X.columns
    assert "vix_chg_1d" in X.columns
    assert not X["spy_ret_15m"].isna().any()
    assert not X["vix_chg_1d"].isna().all()


def test_market_context_requires_spy_and_vol_proxy() -> None:
    with pytest.raises(ValueError, match="volatility proxy"):
        ExperimentConfig(
            symbol="RKLB",
            data_start=date(2024, 1, 1),
            data_end=date(2024, 6, 1),
            context_ingest={"symbols": [{"symbol": "SPY", "interval": "1min"}]},
            features={
                "log_entry_close": True,
                "market_context": True,
            },
        )


def test_g3_warmup_uses_spy_return_bars() -> None:
    fc = FeatureConfig(
        log_entry_close=True,
        market_context=True,
        market_spy_return_bars=15,
    )
    assert feature_warmup_bars(fc) == 15
