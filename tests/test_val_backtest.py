"""Phase I1 validation backtest."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sparkles.backtest.pnl import (
    max_drawdown,
    realized_return_fraction,
)
from sparkles.backtest.val_backtest import (
    build_trade_rows,
    run_val_backtest,
    summarize_trades,
)
from sparkles.config.schema import ExperimentConfig


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "exchange_timezone": "America/New_York",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
        "train_start": date(2024, 6, 3),
        "train_end": date(2024, 6, 4),
        "val_start": date(2024, 6, 5),
        "val_end": date(2024, 6, 7),
        "label_entry_stride": 1,
        "max_day_trades": 1,
        "rolling_business_days": 5,
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_realized_return_fraction_barrier_moves() -> None:
    assert realized_return_fraction(
        "take_profit",
        tp_move_effective=0.05,
        sl_move=0.02,
        entry_close=100.0,
    ) == pytest.approx(0.05)
    assert realized_return_fraction(
        "stop_loss",
        tp_move_effective=0.05,
        sl_move=0.02,
        entry_close=100.0,
    ) == pytest.approx(-0.02)


def test_realized_return_vertical_uses_exit_close() -> None:
    assert realized_return_fraction(
        "vertical",
        tp_move_effective=0.05,
        sl_move=0.02,
        entry_close=100.0,
        exit_close=101.0,
    ) == pytest.approx(0.01)


def test_max_drawdown() -> None:
    s = pd.Series([0.01, 0.03, -0.01, 0.02]).cumsum()
    assert max_drawdown(s) == pytest.approx(-0.01)


def test_build_trade_rows_tp_sl_and_cap(tmp_path: Path) -> None:
    tz = "America/New_York"
    cfg = _cfg(max_day_trades=1, rolling_business_days=5)
    t0 = pd.Timestamp("2024-06-05 10:00", tz=tz)
    t1 = pd.Timestamp("2024-06-05 10:05", tz=tz)
    signals = pd.DataFrame(
        {
            "entry_time": [t0, t1],
            "session_date": [date(2024, 6, 5), date(2024, 6, 5)],
            "y_pred": ["take_profit", "take_profit"],
            "y_true": ["take_profit", "stop_loss"],
            "proba_take_profit": [0.6, 0.55],
        },
    )
    labels = pd.DataFrame(
        {
            "entry_close": [100.0, 100.0],
            "tp_move_effective": [0.05, 0.05],
            "sl_move": [0.02, 0.02],
            "bars_forward": [3, 2],
        },
        index=pd.DatetimeIndex([t0, t1], name="entry_time"),
    )
    idx = pd.date_range("2024-06-05 09:30", periods=20, freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        },
        index=idx,
    )
    trades = build_trade_rows(
        signals,
        labels,
        ohlcv,
        cfg,
        enforce_day_trade_cap=True,
    )
    assert len(trades) == 2
    assert bool(trades.iloc[0]["taken"])
    assert trades.iloc[0]["pnl_fraction"] == pytest.approx(0.05)
    assert not bool(trades.iloc[1]["taken"])
    assert trades.iloc[1]["blocked_reason"] == "day_trade_cap"


def test_run_val_backtest_writes_artifacts(tmp_path: Path) -> None:
    tz = "America/New_York"
    cfg = _cfg()
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    t0 = pd.Timestamp("2024-06-05 10:00", tz=tz)
    labels = pd.DataFrame(
        {
            "entry_close": [100.0],
            "tp_move_effective": [0.04],
            "sl_move": [0.02],
            "bars_forward": [1],
            "barrier_outcome": ["take_profit"],
        },
        index=pd.DatetimeIndex([t0], name="entry_time"),
    )
    labels.to_parquet(cache / "T_labeled_2024-06-01_2024-06-10_s1.parquet")
    idx = pd.date_range("2024-06-05 09:30", periods=10, freq="1min", tz=tz)
    ohlcv = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        },
        index=idx,
    )
    ohlcv.to_parquet(cache / "T_1min_2024-06-01_2024-06-10.parquet")

    run_dir = tmp_path / "artifacts" / "T" / "run1"
    run_dir.mkdir(parents=True)
    preds = pd.DataFrame(
        {
            "entry_time": [t0.tz_localize(None)],
            "session_date": [date(2024, 6, 5)],
            "split": ["val"],
            "y_true": ["take_profit"],
            "y_pred": ["take_profit"],
            "proba_take_profit": [0.7],
        },
    )
    preds.to_parquet(run_dir / "predictions.parquet", index=False)

    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"cache_dir": "data/cache", "artifacts_dir": "artifacts"},
            ),
        },
    )
    summary, trades = run_val_backtest(
        cfg,
        run_dir,
        split="val",
        enforce_day_trade_cap=False,
        base_dir=tmp_path,
    )
    assert (run_dir / "backtest_summary.json").is_file()
    assert (run_dir / "backtest_trades.parquet").is_file()
    assert summary["n_signals"] == 1
    assert summary["n_trades_taken"] == 1
    assert summary["gross_return_sum"] == pytest.approx(0.04)
    assert len(trades) == 1


def test_summarize_trades_empty_signals() -> None:
    preds = pd.DataFrame({"split": ["val"]})
    trades = pd.DataFrame(
        columns=[
            "taken",
            "blocked_reason",
            "y_true",
            "pnl_fraction",
        ],
    )
    summary = summarize_trades(
        preds,
        trades,
        policy="argmax_take_profit",
        split="val",
        enforce_day_trade_cap=True,
    )
    assert summary["n_signals"] == 0
    assert summary["gross_return_sum"] == 0.0
