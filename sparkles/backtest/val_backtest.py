"""Validation backtest from exported predictions + labeled cache (Phase I1)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from sparkles.backtest.pnl import (
    exit_close_at_bars_forward,
    max_drawdown,
    realized_return_fraction,
)
from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.features.volatility import ensure_exchange_tz_index
from sparkles.labels.triple_barrier import labeled_parquet_path
from sparkles.risk.day_trade_ledger import DayTradeLedger

logger = logging.getLogger(__name__)

BACKTEST_ASSUMPTIONS: tuple[str, ...] = (
    "No slippage, fees, or borrow costs.",
    "Each signal is a full-size hypothetical long; overlapping entries are allowed.",
    "take_profit / stop_loss payoffs use label geometry (tp_move_effective, sl_move).",
    "vertical / end_of_data payoffs use OHLCV close at bars_forward from entry.",
    "Day-trade cap (when enabled) skips signals chronologically when the rolling "
    "window is full; ledger uses same-session round trips only.",
)

POLICY_ARGMAX_TAKE_PROFIT = "argmax_take_profit"
POLICY_THRESHOLD_TAKE_PROFIT = "proba_threshold_take_profit"
PROBA_TAKE_PROFIT_COL = "proba_take_profit"


@dataclass
class BacktestContext:
    """Cached inputs for one backtest run (Phase I1/I2)."""

    predictions: pd.DataFrame
    labels: pd.DataFrame
    ohlcv: pd.DataFrame
    run_dir: Path
    split: str


def resolve_run_dir(
    cfg: ExperimentConfig,
    run_id: str | None,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Artifact folder containing ``predictions.parquet``."""
    root = Path.cwd() if base_dir is None else base_dir
    sym_dir = root / cfg.paths.artifacts_dir / cfg.symbol.upper()
    if not sym_dir.is_dir():
        raise FileNotFoundError(f"No artifact directory for symbol: {sym_dir}")

    if run_id:
        run_dir = sym_dir / run_id
        if not (run_dir / "predictions.parquet").is_file():
            raise FileNotFoundError(f"No predictions.parquet in {run_dir}")
        return run_dir

    subdirs = sorted(
        (p for p in sym_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in subdirs:
        if (run_dir / "predictions.parquet").is_file():
            return run_dir
    raise FileNotFoundError(
        "No run with predictions.parquet found; train with "
        "export_predictions: val or all.",
    )


def resolve_backtest_policy(
    cfg: ExperimentConfig,
    *,
    tp_threshold: float | None = None,
) -> tuple[str, float | None]:
    """Pick entry policy: explicit CLI threshold, YAML opt-in, or argmax default."""
    if tp_threshold is not None:
        return POLICY_THRESHOLD_TAKE_PROFIT, tp_threshold
    yaml_threshold = cfg.train.entry_threshold_take_profit
    if yaml_threshold is not None:
        return POLICY_THRESHOLD_TAKE_PROFIT, yaml_threshold
    return POLICY_ARGMAX_TAKE_PROFIT, None


def _signal_mask(
    predictions: pd.DataFrame,
    policy: str,
    *,
    threshold: float | None = None,
) -> pd.Series:
    if policy == POLICY_ARGMAX_TAKE_PROFIT:
        return predictions["y_pred"].astype(str) == "take_profit"
    if policy == POLICY_THRESHOLD_TAKE_PROFIT:
        if threshold is None:
            raise ValueError(
                "threshold required for proba_threshold_take_profit policy",
            )
        if PROBA_TAKE_PROFIT_COL not in predictions.columns:
            raise ValueError(
                f"predictions.parquet missing {PROBA_TAKE_PROFIT_COL!r}; "
                "train with a model that supports predict_proba",
            )
        return predictions[PROBA_TAKE_PROFIT_COL].astype(float) >= threshold
    raise ValueError(f"Unknown backtest policy: {policy!r}")


def signal_classification_metrics(
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
) -> dict[str, float]:
    """Precision/recall for take_profit on val rows at the current entry rule."""
    n_val_tp = int((predictions["y_true"].astype(str) == "take_profit").sum())
    n_signals = int(len(signals))
    if n_signals:
        n_signal_tp = int((signals["y_true"].astype(str) == "take_profit").sum())
    else:
        n_signal_tp = 0
    precision = float(n_signal_tp / n_signals) if n_signals else 0.0
    recall = float(n_signal_tp / n_val_tp) if n_val_tp else 0.0
    return {
        "precision_take_profit": precision,
        "recall_take_profit": recall,
        "n_val_take_profit": float(n_val_tp),
        "n_signal_take_profit": float(n_signal_tp),
    }


def _align_entry_times(
    entry_times: pd.Series,
    exchange_timezone: str,
) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(entry_times)
    return ensure_exchange_tz_index(ix, exchange_timezone)


def build_trade_rows(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    enforce_day_trade_cap: bool,
) -> pd.DataFrame:
    """One row per model signal with PnL proxy and take/skip decision."""
    tz = cfg.exchange_timezone
    label_cols = [
        "entry_close",
        "tp_move_effective",
        "sl_move",
        "bars_forward",
    ]
    merged = signals.merge(
        labels[label_cols],
        left_on="entry_time",
        right_index=True,
        how="left",
    )
    missing = merged["entry_close"].isna()
    if missing.any():
        n = int(missing.sum())
        logger.warning("Dropped %s signals with no labeled-row match", n)
        merged = merged.loc[~missing].copy()

    ledger = DayTradeLedger(
        max_day_trades=cfg.max_day_trades,
        rolling_business_days=cfg.rolling_business_days,
    )
    rows: list[dict[str, Any]] = []
    merged = merged.sort_values("entry_time")

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "session_date",
                "y_true",
                "y_pred",
                "proba_take_profit",
                "taken",
                "blocked_reason",
                "pnl_fraction",
                "exit_session_date",
            ],
        )

    for _, row in merged.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        outcome = str(row["y_true"])
        entry_close = float(row["entry_close"])
        tp_move = float(row["tp_move_effective"])
        sl_move = float(row["sl_move"])
        bars_forward = int(row["bars_forward"])
        entry_session = row["session_date"]
        if isinstance(entry_session, str):
            entry_session = date.fromisoformat(entry_session)
        elif not isinstance(entry_session, date):
            entry_session = pd.Timestamp(entry_session).date()

        exit_close: float | None = None
        exit_session: date | None = None
        if outcome in ("vertical", "end_of_data"):
            exit_close, exit_session = exit_close_at_bars_forward(
                ohlcv,
                entry_time,
                bars_forward,
                tz,
            )
            if exit_close is None:
                rows.append(
                    {
                        "entry_time": entry_time,
                        "session_date": entry_session,
                        "y_true": outcome,
                        "y_pred": str(row["y_pred"]),
                        "proba_take_profit": float(
                            row.get("proba_take_profit", float("nan")),
                        ),
                        "taken": False,
                        "blocked_reason": "missing_ohlcv_exit",
                        "pnl_fraction": float("nan"),
                        "exit_session_date": None,
                    },
                )
                continue

        pnl = realized_return_fraction(
            outcome,
            tp_move_effective=tp_move,
            sl_move=sl_move,
            entry_close=entry_close,
            exit_close=exit_close,
        )

        if exit_session is None:
            if outcome in ("take_profit", "stop_loss"):
                exit_session = entry_session
            else:
                _, exit_session = exit_close_at_bars_forward(
                    ohlcv,
                    entry_time,
                    bars_forward,
                    tz,
                )

        blocked_reason: str | None = None
        taken = True
        if enforce_day_trade_cap and exit_session == entry_session:
            if not ledger.can_add_day_trade(entry_session):
                taken = False
                blocked_reason = "day_trade_cap"

        if taken and enforce_day_trade_cap and exit_session == entry_session:
            ledger.record(entry_session)

        rows.append(
            {
                "entry_time": entry_time,
                "session_date": entry_session,
                "y_true": outcome,
                "y_pred": str(row["y_pred"]),
                "proba_take_profit": float(row.get("proba_take_profit", float("nan"))),
                "taken": taken,
                "blocked_reason": blocked_reason,
                "pnl_fraction": pnl if taken else float("nan"),
                "exit_session_date": exit_session,
            },
        )

    return pd.DataFrame(rows)


def summarize_trades(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    policy: str,
    split: str,
    enforce_day_trade_cap: bool,
    tp_threshold: float | None = None,
    classification: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable summary for ``backtest_summary.json``."""
    if trades.empty or "taken" not in trades.columns:
        base: dict[str, Any] = {
            "policy": policy,
            "split": split,
            "enforce_day_trade_cap": enforce_day_trade_cap,
            "n_predictions": int(len(predictions)),
            "n_signals": 0,
            "n_trades_taken": 0,
            "n_trades_blocked_day_trade_cap": 0,
            "n_trades_skipped_missing_label": 0,
            "outcome_mix_on_taken": {},
            "hit_rate_take_profit_on_taken": 0.0,
            "gross_return_sum": 0.0,
            "gross_return_mean": 0.0,
            "max_drawdown_cumulative_sum": 0.0,
            "assumptions": list(BACKTEST_ASSUMPTIONS),
        }
        if tp_threshold is not None:
            base["tp_threshold"] = tp_threshold
        if classification is not None:
            base["precision_take_profit"] = classification["precision_take_profit"]
            base["recall_take_profit"] = classification["recall_take_profit"]
        return base

    taken = trades.loc[trades["taken"]].copy()
    n_signals = int(len(trades))
    n_taken = int(len(taken))
    blocked = trades.loc[~trades["taken"]]
    n_blocked_cap = int((blocked["blocked_reason"] == "day_trade_cap").sum())

    outcome_mix: dict[str, int] = {}
    hit_rate_tp = 0.0
    gross_sum = 0.0
    gross_mean = 0.0
    max_dd = 0.0

    if n_taken:
        vc = taken["y_true"].astype(str).value_counts()
        outcome_mix = {str(k): int(v) for k, v in vc.items()}
        hit_rate_tp = float((taken["y_true"] == "take_profit").mean())
        pnl = taken["pnl_fraction"].astype(float)
        gross_sum = float(pnl.sum())
        gross_mean = float(pnl.mean())
        cumulative = pnl.cumsum()
        max_dd = max_drawdown(cumulative)

    summary: dict[str, Any] = {
        "policy": policy,
        "split": split,
        "enforce_day_trade_cap": enforce_day_trade_cap,
        "n_predictions": int(len(predictions)),
        "n_signals": n_signals,
        "n_trades_taken": n_taken,
        "n_trades_blocked_day_trade_cap": n_blocked_cap,
        "n_trades_skipped_missing_label": int(
            (trades["blocked_reason"] == "missing_ohlcv_exit").sum(),
        ),
        "outcome_mix_on_taken": outcome_mix,
        "hit_rate_take_profit_on_taken": hit_rate_tp,
        "gross_return_sum": gross_sum,
        "gross_return_mean": gross_mean,
        "max_drawdown_cumulative_sum": max_dd,
        "assumptions": list(BACKTEST_ASSUMPTIONS),
    }
    if tp_threshold is not None:
        summary["tp_threshold"] = tp_threshold
    if classification is not None:
        summary["precision_take_profit"] = classification["precision_take_profit"]
        summary["recall_take_profit"] = classification["recall_take_profit"]
    return summary


def format_backtest_report(summary: dict[str, Any]) -> str:
    """Human-readable multi-line report for CLI."""
    policy_line = (
        f"policy={summary['policy']}  split={summary['split']}  "
        f"day_trade_cap={summary['enforce_day_trade_cap']}"
    )
    if summary.get("tp_threshold") is not None:
        policy_line += f"  tp_threshold={summary['tp_threshold']:.4f}"
    lines = [
        policy_line,
        f"predictions={summary['n_predictions']}  signals={summary['n_signals']}  "
        f"taken={summary['n_trades_taken']}  "
        f"blocked_cap={summary['n_trades_blocked_day_trade_cap']}",
        f"hit_rate_take_profit={summary['hit_rate_take_profit_on_taken']:.4f}  "
        f"gross_return_sum={summary['gross_return_sum']:.4f}  "
        f"gross_return_mean={summary['gross_return_mean']:.4f}  "
        f"max_drawdown={summary['max_drawdown_cumulative_sum']:.4f}",
    ]
    if "precision_take_profit" in summary:
        lines.append(
            f"precision_take_profit={summary['precision_take_profit']:.4f}  "
            f"recall_take_profit={summary['recall_take_profit']:.4f}",
        )
    mix = summary.get("outcome_mix_on_taken") or {}
    if mix:
        parts = "  ".join(f"{k}={v}" for k, v in sorted(mix.items()))
        lines.append(f"outcomes_on_taken: {parts}")
    return "\n".join(lines)


def load_backtest_context(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    split: str = "val",
    base_dir: Path | None = None,
) -> BacktestContext:
    """Load predictions, labels, and OHLCV for backtest / sweep."""
    root = Path.cwd() if base_dir is None else base_dir
    pred_path = run_dir / "predictions.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing {pred_path}")

    predictions = pd.read_parquet(pred_path)
    if split not in ("val", "train", "all"):
        raise ValueError("--split must be val, train, or all")
    if split != "all":
        predictions = predictions.loc[predictions["split"] == split].copy()
    if predictions.empty:
        raise ValueError(f"No prediction rows for split={split!r}")

    label_path = labeled_parquet_path(cfg, base_dir=root)
    if not label_path.is_file():
        raise FileNotFoundError(f"Labeled Parquet not found: {label_path}")

    labels = pd.read_parquet(label_path)
    if labels.index.name != "entry_time":
        if "entry_time" in labels.columns:
            labels = labels.set_index("entry_time")
        else:
            raise ValueError("Labeled Parquet must be indexed by entry_time")
    labels.index = _align_entry_times(pd.Series(labels.index), cfg.exchange_timezone)

    ohlcv_path = parquet_cache_path(cfg, base_dir=root)
    if not ohlcv_path.is_file():
        raise FileNotFoundError(f"Ingest Parquet not found: {ohlcv_path}")
    ohlcv = pd.read_parquet(ohlcv_path)

    return BacktestContext(
        predictions=predictions,
        labels=labels,
        ohlcv=ohlcv,
        run_dir=run_dir,
        split=split,
    )


def run_val_backtest(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    split: str = "val",
    policy: str | None = None,
    tp_threshold: float | None = None,
    enforce_day_trade_cap: bool = True,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run backtest; write ``backtest_summary.json`` and ``backtest_trades.parquet``."""
    resolved_policy, resolved_threshold = resolve_backtest_policy(
        cfg,
        tp_threshold=tp_threshold,
    )
    if policy is not None:
        resolved_policy = policy
        if resolved_policy == POLICY_ARGMAX_TAKE_PROFIT:
            resolved_threshold = None

    ctx = load_backtest_context(
        cfg,
        run_dir,
        split=split,
        base_dir=base_dir,
    )
    mask = _signal_mask(
        ctx.predictions,
        resolved_policy,
        threshold=resolved_threshold,
    )
    signals = ctx.predictions.loc[mask].copy()
    signals["entry_time"] = _align_entry_times(
        signals["entry_time"],
        cfg.exchange_timezone,
    )
    classification = signal_classification_metrics(ctx.predictions, signals)
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
        policy=resolved_policy,
        split=split,
        enforce_day_trade_cap=enforce_day_trade_cap,
        tp_threshold=resolved_threshold,
        classification=classification,
    )
    summary["run_id"] = run_dir.name
    summary["predictions_file"] = "predictions.parquet"

    summary_path = run_dir / "backtest_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trades_path = run_dir / "backtest_trades.parquet"
    trades.to_parquet(trades_path, index=False, engine="pyarrow")
    summary["trades_file"] = trades_path.name
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", summary_path, trades_path)
    return summary, trades
