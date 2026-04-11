"""Training entrypoint: time split, fit, save artifact (Iteration 6).

Edit this file for day-to-day model experiments. Stable hyperparameters also
live under ``model:`` in configs/experiments/*.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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

    sk_cw = (
        resolve_logistic_class_weight(cfg.model, le)
        if cfg.model.type == "logistic_regression"
        else None
    )
    est = build_estimator(cfg, logistic_class_weight=sk_cw)
    if cfg.model.type == "xgboost_classifier":
        sw = xgboost_fit_sample_weight(cfg.model, le, y_tr_e)
        fit_kw: dict[str, Any] = {}
        if sw is not None:
            fit_kw["sample_weight"] = sw
        est.fit(X_tr.values, y_tr_e, **fit_kw)
    else:
        est.fit(X_tr.values, y_tr_e)

    pred_tr = est.predict(X_tr.values)
    pred_va = est.predict(X_va.values)
    names = [str(x) for x in le.classes_]
    feat_dump = cfg.features.model_dump()
    metrics: dict[str, Any] = {
        "model_type": cfg.model.type,
        "train_accuracy": float(accuracy_score(y_tr_e, pred_tr)),
        "val_accuracy": float(accuracy_score(y_va_e, pred_va)),
        "train_n": int(len(X_tr)),
        "val_n": int(len(X_va)),
        "classes": names,
        "features": feat_dump,
        "classification_report_val": classification_report(
            y_va_e,
            pred_va,
            labels=list(range(len(names))),
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
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
        "predictions_export": tc.export_predictions,
        "experiment_config": experiment_snapshot,
    }
    append_experiment_record(art_root, log_payload)
    logger.info("Saved run to %s", out_dir)
    return out_dir
