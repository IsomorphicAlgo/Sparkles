"""Estimator factory: ``model.type`` → sklearn-compatible classifier (Phase C)."""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sparkles.config.schema import ExperimentConfig, ModelConfig

_XGB_INSTALL_HINT = (
    "model.type 'xgboost_classifier' requires the optional [ml] extra. "
    'Install with: pip install -e ".[ml]"'
)


def _xgboost_classifier_type() -> type[Any]:
    try:
        mod = importlib.import_module("xgboost")
    except ImportError as e:
        raise ImportError(_XGB_INSTALL_HINT) from e
    return cast(type[Any], mod.XGBClassifier)


def resolve_logistic_class_weight(
    mc: ModelConfig,
    le: LabelEncoder,
) -> str | dict[int, float] | None:
    """Map YAML class_weight (per-class names) to sklearn integer labels."""
    cw = mc.class_weight
    if cw is None or cw == "balanced":
        return cw
    out: dict[int, float] = {}
    classes = np.asarray(le.classes_)
    for name, w in cw.items():
        hits = np.nonzero(classes == str(name))[0]
        if hits.size == 0:
            raise ValueError(
                f"class_weight key {name!r} not in training classes {classes.tolist()}",
            )
        out[int(hits[0])] = float(w)
    return out


def xgboost_fit_sample_weight(
    mc: ModelConfig,
    le: LabelEncoder,
    y_tr_e: Any,
) -> Any:
    """Optional per-row weights for XGBClassifier.fit (``class_weight`` in YAML)."""
    if mc.class_weight is None:
        return None
    if mc.class_weight == "balanced":
        return cast(Any, compute_sample_weight("balanced", y_tr_e))
    resolved = resolve_logistic_class_weight(mc, le)
    assert isinstance(resolved, dict)
    return cast(Any, compute_sample_weight(resolved, y_tr_e))


def build_estimator(
    cfg: ExperimentConfig,
    *,
    logistic_class_weight: str | dict[int, float] | None = None,
) -> Any:
    """Return an unfitted classifier for ``cfg.model.type``."""
    mc = cfg.model
    if mc.type == "logistic_regression":
        return LogisticRegression(
            C=mc.logistic_c,
            max_iter=mc.max_iter,
            random_state=mc.random_seed,
            solver=mc.solver,
            tol=mc.tol,
            class_weight=logistic_class_weight,
        )
    if mc.type == "xgboost_classifier":
        XGBClassifier = _xgboost_classifier_type()
        return XGBClassifier(
            n_estimators=mc.xgb_n_estimators,
            max_depth=mc.xgb_max_depth,
            learning_rate=mc.xgb_learning_rate,
            subsample=mc.xgb_subsample,
            colsample_bytree=mc.xgb_colsample_bytree,
            min_child_weight=mc.xgb_min_child_weight,
            random_state=mc.random_seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
            verbosity=0,
        )
    raise ValueError(
        f"Unsupported model.type {mc.type!r}. "
        "Use 'logistic_regression' or 'xgboost_classifier'.",
    )
