"""TP probability threshold sweep on val predictions (Phase I2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sparkles.backtest.val_backtest import (
    POLICY_THRESHOLD_TAKE_PROFIT,
    BacktestContext,
    _align_entry_times,
    build_trade_rows,
    load_backtest_context,
    signal_classification_metrics,
    summarize_trades,
)
from sparkles.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


def default_threshold_grid(
    *,
    start: float = 0.05,
    stop: float = 0.96,
    step: float = 0.05,
) -> list[float]:
    """Evenly spaced thresholds in ``(0, 1]`` for sweep tables."""
    if step <= 0 or start <= 0 or stop > 1.0:
        raise ValueError("Invalid threshold grid bounds")
    vals = np.arange(start, stop, step)
    rounded = [round(float(v), 4) for v in vals]
    if not rounded or rounded[-1] < stop - step:
        rounded.append(round(min(stop, 0.99), 4))
    return sorted(set(rounded))


def suggest_threshold(
    sweep: pd.DataFrame,
    *,
    min_signals: int = 5,
) -> dict[str, Any] | None:
    """Heuristic knee: best TP precision with at least ``min_signals`` entries."""
    if sweep.empty:
        return None
    eligible = sweep.loc[sweep["n_signals"] >= min_signals].copy()
    if eligible.empty:
        return None
    best = eligible.sort_values(
        ["precision_take_profit", "gross_return_sum", "n_signals"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "n_signals": int(best["n_signals"]),
        "precision_take_profit": float(best["precision_take_profit"]),
        "recall_take_profit": float(best["recall_take_profit"]),
        "gross_return_sum": float(best["gross_return_sum"]),
        "rationale": (
            f"Highest precision_take_profit among thresholds with "
            f"n_signals>={min_signals}"
        ),
    }


def _one_threshold_row(
    ctx: BacktestContext,
    cfg: ExperimentConfig,
    threshold: float,
    *,
    enforce_day_trade_cap: bool,
) -> dict[str, Any]:
    from sparkles.backtest.val_backtest import _signal_mask

    mask = _signal_mask(
        ctx.predictions,
        POLICY_THRESHOLD_TAKE_PROFIT,
        threshold=threshold,
    )
    signals = ctx.predictions.loc[mask].copy()
    signals["entry_time"] = _align_entry_times(
        signals["entry_time"],
        cfg.exchange_timezone,
    )
    cls = signal_classification_metrics(ctx.predictions, signals)
    trades = build_trade_rows(
        signals,
        ctx.labels,
        ctx.ohlcv,
        cfg,
        enforce_day_trade_cap=enforce_day_trade_cap,
    )
    summary = summarize_trades(
        ctx.predictions,
        trades,
        policy=POLICY_THRESHOLD_TAKE_PROFIT,
        split=ctx.split,
        enforce_day_trade_cap=enforce_day_trade_cap,
        tp_threshold=threshold,
    )
    row = {
        "threshold": threshold,
        "n_signals": summary["n_signals"],
        "n_trades_taken": summary["n_trades_taken"],
        "n_trades_blocked_day_trade_cap": summary["n_trades_blocked_day_trade_cap"],
        "precision_take_profit": cls["precision_take_profit"],
        "recall_take_profit": cls["recall_take_profit"],
        "hit_rate_take_profit_on_taken": summary["hit_rate_take_profit_on_taken"],
        "gross_return_sum": summary["gross_return_sum"],
        "gross_return_mean": summary["gross_return_mean"],
        "max_drawdown_cumulative_sum": summary["max_drawdown_cumulative_sum"],
    }
    return row


def run_threshold_sweep(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    split: str = "val",
    thresholds: list[float] | None = None,
    sweep_step: float = 0.05,
    enforce_day_trade_cap: bool = True,
    min_signals_for_suggestion: int = 5,
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sweep ``proba_take_profit`` thresholds; write CSV + JSON in ``run_dir``."""
    grid = thresholds or default_threshold_grid(step=sweep_step)
    ctx = load_backtest_context(
        cfg,
        run_dir,
        split=split,
        base_dir=base_dir,
    )
    rows = [
        _one_threshold_row(
            ctx,
            cfg,
            t,
            enforce_day_trade_cap=enforce_day_trade_cap,
        )
        for t in grid
    ]
    sweep_df = pd.DataFrame(rows)
    suggestion = suggest_threshold(
        sweep_df,
        min_signals=min_signals_for_suggestion,
    )
    payload: dict[str, Any] = {
        "run_id": run_dir.name,
        "split": split,
        "enforce_day_trade_cap": enforce_day_trade_cap,
        "thresholds": grid,
        "min_signals_for_suggestion": min_signals_for_suggestion,
        "suggested_threshold": suggestion,
        "rows": rows,
    }
    csv_path = run_dir / "backtest_threshold_sweep.csv"
    json_path = run_dir / "backtest_threshold_sweep.json"
    sweep_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", csv_path, json_path)
    return sweep_df, payload


def format_sweep_report(
    sweep_df: pd.DataFrame,
    payload: dict[str, Any],
) -> str:
    """Compact CLI table for threshold sweep."""
    lines = ["threshold  signals  precision  recall  taken  gross_sum  max_dd"]
    for _, row in sweep_df.iterrows():
        lines.append(
            f"{row['threshold']:>9.2f}  "
            f"{int(row['n_signals']):>7}  "
            f"{row['precision_take_profit']:>9.3f}  "
            f"{row['recall_take_profit']:>6.3f}  "
            f"{int(row['n_trades_taken']):>5}  "
            f"{row['gross_return_sum']:>9.3f}  "
            f"{row['max_drawdown_cumulative_sum']:>6.3f}",
        )
    sug = payload.get("suggested_threshold")
    if sug:
        lines.append(
            f"\nsuggested threshold={sug['threshold']:.2f}  "
            f"precision={sug['precision_take_profit']:.3f}  "
            f"signals={sug['n_signals']}",
        )
    return "\n".join(lines)
