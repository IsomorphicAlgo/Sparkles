"""Training entrypoint: time split, fit, save artifact (Iteration 6).

Edit this file for day-to-day model experiments. Stable hyperparameters also
live under ``model:`` in configs/experiments/*.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.features.dataset import (
    build_feature_matrix,
    train_val_masks_by_session_date,
)
from sparkles.labels.triple_barrier import labeled_parquet_path
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


def build_estimator(cfg: ExperimentConfig) -> LogisticRegression:
    """Return an unfitted sklearn estimator from experiment config."""
    mc = cfg.model
    if mc.type != "logistic_regression":
        raise ValueError(
            f"Unsupported model.type {mc.type!r}; "
            "only 'logistic_regression' in Phase 1",
        )
    return LogisticRegression(
        C=mc.logistic_c,
        max_iter=mc.max_iter,
        random_state=mc.random_seed,
        solver="lbfgs",
    )


def run_train(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    labels: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> Path:
    """Load labeled + ingest Parquet, fit baseline LR, write bundle + metrics.

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

    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_tr)
    known = set(str(x) for x in le.classes_)
    mask_va = y_va.astype(str).isin(known)
    if not bool(mask_va.all()):
        dropped = int((~mask_va).sum())
        logger.warning(
            "Dropping %s val rows with outcome classes unseen in train",
            dropped,
        )
    X_va, y_va = X_va.loc[mask_va], y_va.loc[mask_va]
    if len(X_va) == 0:
        raise ValueError("Val split empty after removing unseen outcome classes.")
    y_va_e = le.transform(y_va.astype(str))

    est = build_estimator(cfg)
    est.fit(X_tr.values, y_tr_e)

    pred_tr = est.predict(X_tr.values)
    pred_va = est.predict(X_va.values)
    names = [str(x) for x in le.classes_]
    metrics: dict[str, Any] = {
        "train_accuracy": float(accuracy_score(y_tr_e, pred_tr)),
        "val_accuracy": float(accuracy_score(y_va_e, pred_va)),
        "train_n": int(len(X_tr)),
        "val_n": int(len(X_va)),
        "classes": names,
        "classification_report_val": classification_report(
            y_va_e,
            pred_va,
            labels=list(range(len(names))),
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }

    run_id = new_run_id()
    out_dir = run_artifact_dir(cfg, run_id, base_dir=base_dir)
    bundle = {
        "estimator": est,
        "label_encoder": le,
        "feature_columns": list(X.columns),
    }
    save_bundle(out_dir / "model_bundle.joblib", bundle)
    save_json(out_dir / "metrics.json", metrics)

    art_root = root / cfg.paths.artifacts_dir
    append_experiment_record(
        art_root,
        {
            "run_id": run_id,
            "symbol": cfg.symbol.upper(),
            "train_n": metrics["train_n"],
            "val_n": metrics["val_n"],
            "val_accuracy": metrics["val_accuracy"],
        },
    )
    logger.info("Saved run to %s", out_dir)
    return out_dir
