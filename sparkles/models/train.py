"""Training entrypoint: time split, fit, save artifact (Iteration 6).

Edit this file for day-to-day model experiments. Stable hyperparameters also
live under ``model:`` in configs/experiments/*.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score

from sparkles.models.evaluation import (
    classification_report_dict,
    f1_macro_weighted,
)
from sklearn.preprocessing import LabelEncoder

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.features.dataset import (
    build_feature_matrix,
    train_val_masks_by_session_date,
)
from sparkles.labels.triple_barrier import labeled_parquet_path
from sparkles.models.estimators import (
    build_estimator,
    resolve_logistic_class_weight,
    xgboost_fit_sample_weight,
)
from sparkles.models.preprocess import (
    build_training_estimator,
    fit_training_estimator,
    predict_values,
)
from sparkles.models.predictions_export import predictions_frame
from sparkles.models.registry import (
    new_run_id,
    run_artifact_dir,
    save_bundle,
    save_json,
)
from sparkles.tracking.experiments import append_experiment_record

logger = logging.getLogger(__name__)

# Optional Python-side overrides while iterating (merged after YAML in future).
DEFAULT_TRAIN_KWARGS: dict[str, Any] = {}


@dataclass(frozen=True)
class PreparedTrainingData:
    """Train/val matrices after split, class filtering, and label encoding."""

    X_tr: pd.DataFrame
    y_tr: pd.Series
    X_va: pd.DataFrame
    y_va: pd.Series
    y_tr_e: Any
    y_va_e: Any
    label_encoder: LabelEncoder
    feature_columns: list[str]
    val_rows_dropped_unseen: int


@dataclass(frozen=True)
class TrainDryRunReport:
    """Pre-flight summary for ``dry_run_train`` (ML expansion Phase E)."""

    symbol: str
    model_type: str
    train_n: int
    val_n: int
    train_class_balance: dict[str, int]
    val_class_balance: dict[str, int]
    feature_columns: list[str]
    features_enabled: dict[str, bool]
    experiment_name: str | None
    notes: str | None
    val_rows_dropped_unseen: int
    ready: bool
    issues: tuple[str, ...] = field(default_factory=tuple)


def _load_labels_ohlcv(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None,
    labels: pd.DataFrame | None,
    ohlcv: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labels is None:
        lpath = labeled_parquet_path(cfg, base_dir=base_dir)
        if not lpath.is_file():
            raise FileNotFoundError(
                f"Labeled Parquet not found: {lpath}. Run `sparkles label` first.",
            )
        labels = pd.read_parquet(lpath)
    if ohlcv is None:
        ipath = parquet_cache_path(cfg, base_dir=base_dir)
        if not ipath.is_file():
            raise FileNotFoundError(
                f"Ingest Parquet not found: {ipath}. Run `sparkles ingest` first.",
            )
        ohlcv = pd.read_parquet(ipath)
    return labels, ohlcv


def prepare_training_data(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    labels: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> PreparedTrainingData:
    """Load data, build features, split, and validate row floors (no fit)."""
    labels, ohlcv = _load_labels_ohlcv(
        cfg,
        base_dir=base_dir,
        labels=labels,
        ohlcv=ohlcv,
    )
    X, y = build_feature_matrix(labels, ohlcv, cfg)
    train_m, val_m = train_val_masks_by_session_date(X.index, cfg)

    X_tr, y_tr = X.loc[train_m], y.loc[train_m]
    X_va, y_va = X.loc[val_m], y.loc[val_m]
    if len(X_tr) == 0 or len(X_va) == 0:
        raise ValueError(
            f"Empty train or val split (train={len(X_tr)}, val={len(X_va)}). "
            "Adjust train_start/train_end and val_start/val_end in the YAML.",
        )

    tc = cfg.train
    if len(X_tr) < tc.min_train_rows:
        raise ValueError(
            f"Train rows {len(X_tr)} below train.min_train_rows={tc.min_train_rows}",
        )

    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_tr)
    known = set(str(x) for x in le.classes_)
    mask_va = y_va.astype(str).isin(known)
    dropped = 0
    if not tc.drop_val_unseen_classes:
        if not bool(mask_va.all()):
            bad = sorted(set(y_va.loc[~mask_va].astype(str)))
            raise ValueError(
                "Val split contains outcome classes not seen in train "
                f"{bad!r}. Enable train.drop_val_unseen_classes or fix date ranges.",
            )
    else:
        if not bool(mask_va.all()):
            dropped = int((~mask_va).sum())
            logger.warning(
                "Dropping %s val rows with outcome classes unseen in train",
                dropped,
            )
        X_va, y_va = X_va.loc[mask_va], y_va.loc[mask_va]
    if len(X_va) == 0:
        raise ValueError("Val split empty after removing unseen outcome classes.")
    if len(X_va) < tc.min_val_rows:
        raise ValueError(
            f"Val rows {len(X_va)} below train.min_val_rows={tc.min_val_rows}",
        )
    y_va_e = le.transform(y_va.astype(str))

    return PreparedTrainingData(
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        y_tr_e=y_tr_e,
        y_va_e=y_va_e,
        label_encoder=le,
        feature_columns=list(X.columns),
        val_rows_dropped_unseen=dropped,
    )


def dry_run_train(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    labels: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> TrainDryRunReport:
    """Summarize row counts, class balance, and features without fitting."""
    tc = cfg.train
    try:
        prep = prepare_training_data(
            cfg,
            base_dir=base_dir,
            labels=labels,
            ohlcv=ohlcv,
        )
        train_bal = prep.y_tr.astype(str).value_counts().sort_index().to_dict()
        val_bal = prep.y_va.astype(str).value_counts().sort_index().to_dict()
        return TrainDryRunReport(
            symbol=cfg.symbol.upper(),
            model_type=cfg.model.type,
            train_n=len(prep.X_tr),
            val_n=len(prep.X_va),
            train_class_balance={str(k): int(v) for k, v in train_bal.items()},
            val_class_balance={str(k): int(v) for k, v in val_bal.items()},
            feature_columns=prep.feature_columns,
            features_enabled=cfg.features.model_dump(),
            experiment_name=tc.experiment_name,
            notes=tc.notes,
            val_rows_dropped_unseen=prep.val_rows_dropped_unseen,
            ready=True,
            issues=(),
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        return TrainDryRunReport(
            symbol=cfg.symbol.upper(),
            model_type=cfg.model.type,
            train_n=0,
            val_n=0,
            train_class_balance={},
            val_class_balance={},
            feature_columns=[],
            features_enabled=cfg.features.model_dump(),
            experiment_name=tc.experiment_name,
            notes=tc.notes,
            val_rows_dropped_unseen=0,
            ready=False,
            issues=(str(e),),
        )


def format_dry_run_report(report: TrainDryRunReport) -> str:
    """Human-readable multi-line summary for CLI."""
    lines = [
        f"symbol={report.symbol}  model_type={report.model_type}  ready={report.ready}",
        f"train_n={report.train_n}  val_n={report.val_n}  "
        f"val_rows_dropped_unseen={report.val_rows_dropped_unseen}",
    ]
    if report.experiment_name:
        lines.append(f"experiment_name={report.experiment_name!r}")
    if report.notes:
        lines.append(f"notes={report.notes!r}")
    if report.train_class_balance:
        lines.append(
            "train_class_balance: "
            + "  ".join(f"{k}={v}" for k, v in report.train_class_balance.items()),
        )
    if report.val_class_balance:
        lines.append(
            "val_class_balance: "
            + "  ".join(f"{k}={v}" for k, v in report.val_class_balance.items()),
        )
    enabled = [k for k, v in report.features_enabled.items() if v is True]
    lines.append(f"features_enabled: {', '.join(enabled) or '(none)'}")
    lines.append(f"feature_columns ({len(report.feature_columns)}): "
                 + ", ".join(report.feature_columns))
    if report.issues:
        lines.append("issues:")
        for issue in report.issues:
            lines.append(f"  - {issue}")
    return "\n".join(lines)


def run_train(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    labels: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> Path:
    """Load labeled + ingest Parquet, fit classifier, write bundle + metrics.

    ``labels`` / ``ohlcv`` override disk loads (used by unit tests).
    """
    root = Path.cwd() if base_dir is None else base_dir
    prep = prepare_training_data(
        cfg,
        base_dir=base_dir,
        labels=labels,
        ohlcv=ohlcv,
    )
    X_tr, y_tr = prep.X_tr, prep.y_tr
    X_va, y_va = prep.X_va, prep.y_va
    y_tr_e, y_va_e = prep.y_tr_e, prep.y_va_e
    le = prep.label_encoder
    X = X_tr  # for feature column list in bundle

    sk_cw = (
        resolve_logistic_class_weight(cfg.model, le)
        if cfg.model.type == "logistic_regression"
        else None
    )
    clf = build_estimator(cfg, logistic_class_weight=sk_cw)
    est = build_training_estimator(cfg, clf)
    sw: Any = None
    if cfg.model.type == "xgboost_classifier":
        sw = xgboost_fit_sample_weight(cfg.model, le, y_tr_e)
    fit_training_estimator(est, X_tr, y_tr_e, sample_weight=sw)

    pred_tr = predict_values(est, X_tr)
    pred_va = predict_values(est, X_va)
    names = [str(x) for x in le.classes_]
    label_ids = list(range(len(names)))
    report_val = classification_report_dict(
        y_va_e,
        pred_va,
        labels=label_ids,
        target_names=names,
    )
    train_f1_macro, train_f1_weighted = f1_macro_weighted(y_tr_e, pred_tr)
    val_f1_macro, val_f1_weighted = f1_macro_weighted(y_va_e, pred_va)
    feat_dump = cfg.features.model_dump()
    tc = cfg.train
    metrics: dict[str, Any] = {
        "model_type": cfg.model.type,
        "train_accuracy": float(accuracy_score(y_tr_e, pred_tr)),
        "val_accuracy": float(accuracy_score(y_va_e, pred_va)),
        "train_f1_macro": train_f1_macro,
        "train_f1_weighted": train_f1_weighted,
        "val_f1_macro": val_f1_macro,
        "val_f1_weighted": val_f1_weighted,
        "train_n": int(len(X_tr)),
        "val_n": int(len(X_va)),
        "classes": names,
        "features": feat_dump,
        "preprocess_scaler": cfg.preprocess.scaler,
        "classification_report_val": report_val,
    }

    pred_parts: list[pd.DataFrame] = []
    if tc.export_predictions in ("val", "all"):
        pred_parts.append(
            predictions_frame(
                X_va,
                y_va,
                y_va_e,
                pred_va,
                "val",
                est,
                le,
                cfg.exchange_timezone,
            ),
        )
    if tc.export_predictions == "all":
        pred_parts.append(
            predictions_frame(
                X_tr,
                y_tr,
                y_tr_e,
                pred_tr,
                "train",
                est,
                le,
                cfg.exchange_timezone,
            ),
        )
    metrics["predictions_export"] = tc.export_predictions
    metrics["predictions_file"] = (
        "predictions.parquet" if pred_parts else None
    )

    run_id = new_run_id()
    out_dir = run_artifact_dir(cfg, run_id, base_dir=base_dir)
    bundle = {
        "estimator": est,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "preprocess_scaler": cfg.preprocess.scaler,
    }
    save_bundle(out_dir / "model_bundle.joblib", bundle)
    save_json(out_dir / "metrics.json", metrics)
    experiment_snapshot: dict[str, Any] = cfg.model_dump(mode="json")
    save_json(out_dir / "experiment_config.json", experiment_snapshot)
    if pred_parts:
        pd.concat(pred_parts, axis=0).to_parquet(
            out_dir / "predictions.parquet",
            index=False,
        )

    art_root = root / cfg.paths.artifacts_dir
    log_payload: dict[str, Any] = {
        "run_id": run_id,
        "symbol": cfg.symbol.upper(),
        "train_n": metrics["train_n"],
        "val_n": metrics["val_n"],
        "val_accuracy": metrics["val_accuracy"],
        "train_accuracy": metrics["train_accuracy"],
        "val_f1_macro": metrics["val_f1_macro"],
        "val_f1_weighted": metrics["val_f1_weighted"],
        "train_f1_macro": metrics["train_f1_macro"],
        "train_f1_weighted": metrics["train_f1_weighted"],
        "label_entry_stride": cfg.label_entry_stride,
        "model_type": cfg.model.type,
        "model_solver": cfg.model.solver
        if cfg.model.type == "logistic_regression"
        else "xgboost",
        "model_xgb_learning_rate": cfg.model.xgb_learning_rate
        if cfg.model.type == "xgboost_classifier"
        else None,
        "model_class_weight": cfg.model.class_weight,
        "train_experiment_name": tc.experiment_name,
        "train_notes": tc.notes,
        "train_min_train_rows": tc.min_train_rows,
        "train_min_val_rows": tc.min_val_rows,
        "train_drop_val_unseen_classes": tc.drop_val_unseen_classes,
        "features": feat_dump,
        "preprocess_scaler": cfg.preprocess.scaler,
        "predictions_export": tc.export_predictions,
        "experiment_config": experiment_snapshot,
    }
    append_experiment_record(art_root, log_payload)
    logger.info("Saved run to %s", out_dir)
    return out_dir
