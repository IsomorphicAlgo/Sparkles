"""Phase I2 TP threshold policy and sweep."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sparkles.backtest.threshold_sweep import (
    default_threshold_grid,
    run_threshold_sweep,
    suggest_threshold,
)
from sparkles.backtest.val_backtest import (
    POLICY_THRESHOLD_TAKE_PROFIT,
    _signal_mask,
    resolve_backtest_policy,
    run_val_backtest,
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
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_signal_mask_threshold() -> None:
    preds = pd.DataFrame(
        {
            "y_pred": ["take_profit", "vertical", "take_profit"],
            "proba_take_profit": [0.2, 0.8, 0.5],
        },
    )
    mask = _signal_mask(
        preds,
        POLICY_THRESHOLD_TAKE_PROFIT,
        threshold=0.45,
    )
    assert mask.tolist() == [False, True, True]


def test_resolve_backtest_policy_yaml_opt_in() -> None:
    cfg = _cfg(train={"entry_threshold_take_profit": 0.33})
    policy, threshold = resolve_backtest_policy(cfg)
    assert policy == POLICY_THRESHOLD_TAKE_PROFIT
    assert threshold == pytest.approx(0.33)


def test_resolve_backtest_policy_cli_overrides_yaml() -> None:
    cfg = _cfg(train={"entry_threshold_take_profit": 0.33})
    policy, threshold = resolve_backtest_policy(cfg, tp_threshold=0.5)
    assert threshold == pytest.approx(0.5)


def test_default_threshold_grid() -> None:
    grid = default_threshold_grid(step=0.25)
    assert 0.05 in grid
    assert 0.3 in grid


def test_suggest_threshold_picks_highest_precision_with_min_signals() -> None:
    sweep = pd.DataFrame(
        {
            "threshold": [0.1, 0.5, 0.9],
            "n_signals": [100, 10, 2],
            "precision_take_profit": [0.1, 0.4, 0.9],
            "recall_take_profit": [0.8, 0.3, 0.05],
            "gross_return_sum": [1.0, 0.5, 0.1],
        },
    )
    sug = suggest_threshold(sweep, min_signals=5)
    assert sug is not None
    assert sug["threshold"] == pytest.approx(0.5)


def test_run_threshold_sweep_writes_artifacts(tmp_path: Path) -> None:
    tz = "America/New_York"
    cfg = _cfg()
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    t0 = pd.Timestamp("2024-06-05 10:00", tz=tz)
    t1 = pd.Timestamp("2024-06-05 10:05", tz=tz)
    labels = pd.DataFrame(
        {
            "entry_close": [100.0, 100.0],
            "tp_move_effective": [0.04, 0.04],
            "sl_move": [0.02, 0.02],
            "bars_forward": [1, 1],
        },
        index=pd.DatetimeIndex([t0, t1], name="entry_time"),
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
            "entry_time": [t0.tz_localize(None), t1.tz_localize(None)],
            "session_date": [date(2024, 6, 5), date(2024, 6, 5)],
            "split": ["val", "val"],
            "y_true": ["take_profit", "stop_loss"],
            "y_pred": ["take_profit", "take_profit"],
            "proba_take_profit": [0.8, 0.2],
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
    sweep_df, payload = run_threshold_sweep(
        cfg,
        run_dir,
        thresholds=[0.1, 0.5],
        enforce_day_trade_cap=False,
        base_dir=tmp_path,
    )
    assert (run_dir / "backtest_threshold_sweep.csv").is_file()
    assert (run_dir / "backtest_threshold_sweep.json").is_file()
    assert len(sweep_df) == 2
    low = sweep_df.loc[sweep_df["threshold"] == 0.1].iloc[0]
    high = sweep_df.loc[sweep_df["threshold"] == 0.5].iloc[0]
    assert int(low["n_signals"]) == 2
    assert int(high["n_signals"]) == 1
    assert payload["suggested_threshold"] is None


def test_run_val_backtest_with_threshold(tmp_path: Path) -> None:
    tz = "America/New_York"
    cfg = _cfg()
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    t0 = pd.Timestamp("2024-06-05 10:00", tz=tz)
    t1 = pd.Timestamp("2024-06-05 10:05", tz=tz)
    labels = pd.DataFrame(
        {
            "entry_close": [100.0, 100.0],
            "tp_move_effective": [0.04, 0.04],
            "sl_move": [0.02, 0.02],
            "bars_forward": [1, 1],
        },
        index=pd.DatetimeIndex([t0, t1], name="entry_time"),
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
            "entry_time": [t0.tz_localize(None), t1.tz_localize(None)],
            "session_date": [date(2024, 6, 5), date(2024, 6, 5)],
            "split": ["val", "val"],
            "y_true": ["take_profit", "stop_loss"],
            "y_pred": ["vertical", "vertical"],
            "proba_take_profit": [0.8, 0.2],
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
    summary, _ = run_val_backtest(
        cfg,
        run_dir,
        tp_threshold=0.5,
        enforce_day_trade_cap=False,
        base_dir=tmp_path,
    )
    assert summary["policy"] == POLICY_THRESHOLD_TAKE_PROFIT
    assert summary["tp_threshold"] == pytest.approx(0.5)
    assert summary["n_signals"] == 1
    assert summary["precision_take_profit"] == pytest.approx(1.0)
