"""AFML-style meta-label spike: filter primary take_profit signals (Phase I3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sparkles.backtest.val_backtest import (
    POLICY_ARGMAX_TAKE_PROFIT,
    POLICY_THRESHOLD_TAKE_PROFIT,
    PROBA_TAKE_PROFIT_COL,
    BacktestContext,
    _align_entry_times,
    build_trade_rows,
    load_backtest_context,
    signal_classification_metrics,
    summarize_trades,
)
from sparkles.config.schema import ExperimentConfig
from sparkles.features.time import entry_session_dates
from sparkles.models.preprocess import load_model_bundle, predict_from_bundle
from sparkles.models.registry import save_bundle, save_json
from sparkles.models.train import prepare_training_data

logger = logging.getLogger(__name__)

POLICY_META_LABEL = "meta_label"
META_FEATURE_COLUMNS: tuple[str, ...] = (
    "proba_stop_loss",
    "proba_take_profit",
    "proba_vertical",
    "max_proba",
)


def resolve_primary_run_dir(
    cfg: ExperimentConfig,
    run_id: str | None,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Artifact folder containing ``model_bundle.joblib``."""
    root = Path.cwd() if base_dir is None else base_dir
    sym_dir = root / cfg.paths.artifacts_dir / cfg.symbol.upper()
    if not sym_dir.is_dir():
        raise FileNotFoundError(f"No artifact directory for symbol: {sym_dir}")

    if run_id:
        run_dir = sym_dir / run_id
        if not (run_dir / "model_bundle.joblib").is_file():
            raise FileNotFoundError(f"No model_bundle.joblib in {run_dir}")
        return run_dir

    subdirs = sorted(
        (p for p in sym_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in subdirs:
        if (run_dir / "model_bundle.joblib").is_file():
            return run_dir
    raise FileNotFoundError(
        "No run with model_bundle.joblib found; run sparkles train first.",
    )


def resolve_primary_threshold(
    cfg: ExperimentConfig,
    *,
    primary_threshold: float | None = None,
) -> float:
    """Primary ``proba_take_profit`` gate shared by I2 meta training and compare."""
    if primary_threshold is not None:
        return primary_threshold
    if cfg.train.entry_threshold_take_profit is not None:
        return cfg.train.entry_threshold_take_profit
    if cfg.train.meta_label_primary_threshold is not None:
        return cfg.train.meta_label_primary_threshold
    return 0.35


def resolve_meta_act_threshold(
    cfg: ExperimentConfig,
    *,
    meta_threshold: float | None = None,
) -> float:
    if meta_threshold is not None:
        return meta_threshold
    return cfg.train.meta_label_act_threshold


def _require_proba(estimator: Any) -> None:
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "Primary estimator must support predict_proba for meta-label spike",
        )


def primary_proba_dataframe(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split: str,
    exchange_timezone: str,
) -> pd.DataFrame:
    """Primary model probabilities aligned to feature rows."""
    _require_proba(bundle["estimator"])
    cols = bundle["feature_columns"]
    x = X[list(cols)].values.astype(np.float64)
    proba = bundle["estimator"].predict_proba(x)
    le = bundle["label_encoder"]
    pred_enc = predict_from_bundle(bundle, X, feature_columns=cols)
    y_pred = le.inverse_transform(pred_enc.astype(np.int64))

    out: dict[str, Any] = {
        "entry_time": X.index,
        "session_date": entry_session_dates(X.index, exchange_timezone).values,
        "split": split,
        "y_true": y.astype(str).values,
        "y_pred": y_pred.astype(str),
    }
    for j, cls in enumerate(le.classes_):
        safe = str(cls).replace(" ", "_")
        out[f"proba_{safe}"] = proba[:, j].astype(np.float64)
    out["max_proba"] = proba.max(axis=1).astype(np.float64)
    if PROBA_TAKE_PROFIT_COL not in out:
        raise ValueError(f"Primary bundle missing {PROBA_TAKE_PROFIT_COL!r} column")
    return pd.DataFrame(out)


def meta_label_targets(y_true: pd.Series) -> np.ndarray[Any, Any]:
    """Binary meta target: 1 iff realized outcome is take_profit."""
    return (y_true.astype(str).values == "take_profit").astype(np.int64)


def build_meta_feature_matrix(proba_df: pd.DataFrame) -> pd.DataFrame:
    """Primary probability features; missing class columns default to 0."""
    out: dict[str, pd.Series] = {}
    for col in META_FEATURE_COLUMNS:
        if col in proba_df.columns:
            out[col] = proba_df[col].astype(np.float64)
        elif col == "max_proba":
            proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
            out[col] = proba_df[proba_cols].max(axis=1).astype(np.float64)
        else:
            out[col] = pd.Series(0.0, index=proba_df.index, dtype=np.float64)
    return pd.DataFrame(out, index=proba_df.index)


def assert_meta_train_within_primary_train(
    meta_train_index: pd.Index,
    primary_train_index: pd.Index,
) -> None:
    """Meta-label fit rows must be a subset of primary train rows (no val leakage)."""
    if len(meta_train_index) == 0:
        return
    extra = meta_train_index.difference(primary_train_index)
    if len(extra):
        raise ValueError(
            f"Meta-label train leaked {len(extra)} rows outside primary train split",
        )


def _load_cfg_for_primary_run(
    cfg: ExperimentConfig,
    primary_run_dir: Path,
) -> ExperimentConfig:
    """Prefer frozen ``experiment_config.json`` from the primary run when present."""
    snap = primary_run_dir / "experiment_config.json"
    if snap.is_file():
        data = json.loads(snap.read_text(encoding="utf-8"))
        return ExperimentConfig(**data)
    return cfg


def train_meta_label(
    cfg: ExperimentConfig,
    primary_run_dir: Path,
    *,
    primary_threshold: float | None = None,
    meta_threshold: float | None = None,
    base_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fit binary meta model on primary-gated train rows; save beside primary run."""
    cfg = _load_cfg_for_primary_run(cfg, primary_run_dir)
    root = Path.cwd() if base_dir is None else base_dir
    bundle_path = primary_run_dir / "model_bundle.joblib"
    bundle = load_model_bundle(bundle_path)
    prep = prepare_training_data(cfg, base_dir=root)

    stored_cols = bundle["feature_columns"]
    if list(prep.feature_columns) != list(stored_cols):
        raise ValueError(
            "Feature columns disagree with primary bundle; "
            "use the same experiment YAML",
        )

    train_proba = primary_proba_dataframe(
        bundle,
        prep.X_tr,
        prep.y_tr,
        split="train",
        exchange_timezone=cfg.exchange_timezone,
    )
    val_proba = primary_proba_dataframe(
        bundle,
        prep.X_va,
        prep.y_va,
        split="val",
        exchange_timezone=cfg.exchange_timezone,
    )

    p_thresh = resolve_primary_threshold(cfg, primary_threshold=primary_threshold)
    m_thresh = resolve_meta_act_threshold(cfg, meta_threshold=meta_threshold)

    train_gate = train_proba[PROBA_TAKE_PROFIT_COL] >= p_thresh
    meta_train_df = train_proba.loc[train_gate].copy()
    assert_meta_train_within_primary_train(
        pd.DatetimeIndex(meta_train_df["entry_time"]),
        prep.X_tr.index,
    )

    if meta_train_df.empty:
        raise ValueError(
            f"No primary-gated train rows at threshold {p_thresh}; "
            "lower primary_threshold",
        )

    y_meta_tr = meta_label_targets(meta_train_df["y_true"])
    X_meta_tr = build_meta_feature_matrix(meta_train_df)
    meta_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=cfg.model.random_seed,
    )
    meta_clf.fit(X_meta_tr.values, y_meta_tr)

    val_gate = val_proba[PROBA_TAKE_PROFIT_COL] >= p_thresh
    meta_val_df = val_proba.loc[val_gate].copy()
    metrics: dict[str, Any] = {
        "primary_run_id": primary_run_dir.name,
        "primary_threshold": p_thresh,
        "meta_act_threshold": m_thresh,
        "meta_feature_columns": list(META_FEATURE_COLUMNS),
        "n_primary_train": int(len(train_proba)),
        "n_meta_train_gated": int(len(meta_train_df)),
        "n_meta_train_positive": int(y_meta_tr.sum()),
        "n_primary_val_gated": int(val_gate.sum()),
    }
    if len(meta_val_df):
        y_meta_va = meta_label_targets(meta_val_df["y_true"])
        pred_va = meta_clf.predict(build_meta_feature_matrix(meta_val_df).values)
        metrics.update(
            {
                "val_accuracy_gated": float(accuracy_score(y_meta_va, pred_va)),
                "val_precision_gated": float(
                    precision_score(y_meta_va, pred_va, zero_division=0.0),
                ),
                "val_recall_gated": float(
                    recall_score(y_meta_va, pred_va, zero_division=0.0),
                ),
            },
        )
    else:
        metrics["val_accuracy_gated"] = None

    meta_bundle = {
        "estimator": meta_clf,
        "feature_columns": list(META_FEATURE_COLUMNS),
        "primary_run_id": primary_run_dir.name,
        "primary_threshold": p_thresh,
        "meta_act_threshold": m_thresh,
    }
    out_path = primary_run_dir / "meta_label_bundle.joblib"
    save_bundle(out_path, meta_bundle)
    save_json(primary_run_dir / "meta_label_metrics.json", metrics)
    logger.info("Saved meta-label bundle to %s", out_path)
    return out_path, metrics


def _signals_from_policy(
    proba_df: pd.DataFrame,
    *,
    policy: str,
    primary_threshold: float,
    meta_bundle: dict[str, Any] | None = None,
    meta_threshold: float = 0.5,
) -> pd.DataFrame:
    if policy == POLICY_ARGMAX_TAKE_PROFIT:
        mask = proba_df["y_pred"].astype(str) == "take_profit"
    elif policy == POLICY_THRESHOLD_TAKE_PROFIT:
        mask = proba_df[PROBA_TAKE_PROFIT_COL] >= primary_threshold
    elif policy == POLICY_META_LABEL:
        if meta_bundle is None:
            raise ValueError("meta_bundle required for meta_label policy")
        primary_mask = proba_df[PROBA_TAKE_PROFIT_COL] >= primary_threshold
        gated = proba_df.loc[primary_mask].copy()
        if gated.empty:
            return gated
        x_meta = build_meta_feature_matrix(gated).values
        meta_proba = meta_bundle["estimator"].predict_proba(x_meta)[:, 1]
        keep = meta_proba >= meta_threshold
        return gated.loc[keep].copy()
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    return proba_df.loc[mask].copy()


def _economics_from_signals(
    signals: pd.DataFrame,
    ctx: BacktestContext,
    cfg: ExperimentConfig,
    *,
    policy: str,
    primary_threshold: float | None,
    enforce_day_trade_cap: bool,
) -> dict[str, Any]:
    if signals.empty:
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
            policy=policy,
            split=ctx.split,
            enforce_day_trade_cap=enforce_day_trade_cap,
            tp_threshold=primary_threshold,
            classification=signal_classification_metrics(ctx.predictions, signals),
        )
        return summary

    aligned = signals.copy()
    aligned["entry_time"] = _align_entry_times(
        aligned["entry_time"],
        cfg.exchange_timezone,
    )
    trades = build_trade_rows(
        aligned,
        ctx.labels,
        ctx.ohlcv,
        cfg,
        enforce_day_trade_cap=enforce_day_trade_cap,
    )
    cls = signal_classification_metrics(ctx.predictions, aligned)
    return summarize_trades(
        ctx.predictions,
        trades,
        policy=policy,
        split=ctx.split,
        enforce_day_trade_cap=enforce_day_trade_cap,
        tp_threshold=primary_threshold,
        classification=cls,
    )


def compare_entry_policies(
    cfg: ExperimentConfig,
    primary_run_dir: Path,
    *,
    primary_threshold: float | None = None,
    meta_threshold: float | None = None,
    enforce_day_trade_cap: bool = True,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare argmax vs I2 threshold vs I3 meta-filter on val (same primary run)."""
    cfg = _load_cfg_for_primary_run(cfg, primary_run_dir)
    root = Path.cwd() if base_dir is None else base_dir
    bundle = load_model_bundle(primary_run_dir / "model_bundle.joblib")
    meta_path = primary_run_dir / "meta_label_bundle.joblib"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}; run `sparkles meta-label train` first",
        )
    meta_bundle = load_model_bundle(meta_path)

    prep = prepare_training_data(cfg, base_dir=root)
    val_proba = primary_proba_dataframe(
        bundle,
        prep.X_va,
        prep.y_va,
        split="val",
        exchange_timezone=cfg.exchange_timezone,
    )
    ctx = load_backtest_context(cfg, primary_run_dir, split="val", base_dir=root)
    ctx = BacktestContext(
        predictions=val_proba,
        labels=ctx.labels,
        ohlcv=ctx.ohlcv,
        run_dir=primary_run_dir,
        split="val",
    )

    p_thresh = resolve_primary_threshold(cfg, primary_threshold=primary_threshold)
    m_thresh = resolve_meta_act_threshold(cfg, meta_threshold=meta_threshold)

    results: dict[str, Any] = {
        "primary_run_id": primary_run_dir.name,
        "primary_threshold": p_thresh,
        "meta_act_threshold": m_thresh,
        "policies": {},
    }

    for policy in (
        POLICY_ARGMAX_TAKE_PROFIT,
        POLICY_THRESHOLD_TAKE_PROFIT,
        POLICY_META_LABEL,
    ):
        signals = _signals_from_policy(
            val_proba,
            policy=policy,
            primary_threshold=p_thresh,
            meta_bundle=meta_bundle,
            meta_threshold=m_thresh,
        )
        summary = _economics_from_signals(
            signals,
            ctx,
            cfg,
            policy=policy,
            primary_threshold=p_thresh if policy != POLICY_ARGMAX_TAKE_PROFIT else None,
            enforce_day_trade_cap=enforce_day_trade_cap,
        )
        results["policies"][policy] = summary

    out_path = primary_run_dir / "meta_label_compare.json"
    out_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def format_compare_report(results: dict[str, Any]) -> str:
    lines = [
        f"primary_run={results['primary_run_id']}  "
        f"primary_threshold={results['primary_threshold']:.2f}  "
        f"meta_act_threshold={results['meta_act_threshold']:.2f}",
        "policy                      signals  precision  recall  taken  gross_sum",
    ]
    for name, summary in results["policies"].items():
        lines.append(
            f"{name:28}  "
            f"{summary['n_signals']:>7}  "
            f"{summary.get('precision_take_profit', 0.0):>9.3f}  "
            f"{summary.get('recall_take_profit', 0.0):>6.3f}  "
            f"{summary['n_trades_taken']:>5}  "
            f"{summary['gross_return_sum']:>9.3f}",
        )
    return "\n".join(lines)
