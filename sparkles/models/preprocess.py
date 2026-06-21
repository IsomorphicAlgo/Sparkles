"""Optional feature scaling in a sklearn Pipeline (ML expansion Phase D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from sparkles.config.schema import ExperimentConfig

ScalerKind = Literal["none", "standard", "robust"]


def build_scaler(kind: ScalerKind) -> StandardScaler | RobustScaler:
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler kind: {kind!r}")


def build_training_estimator(
    cfg: ExperimentConfig,
    classifier: Any,
) -> Any:
    """Return unfitted estimator or Pipeline(scaler, classifier) per config."""
    kind = cfg.preprocess.scaler
    if kind == "none":
        return classifier
    return Pipeline(
        [
            ("scaler", build_scaler(kind)),
            ("clf", classifier),
        ],
    )


def fit_training_estimator(
    estimator: Any,
    X_tr: pd.DataFrame,
    y_tr_e: Any,
    *,
    sample_weight: Any = None,
) -> Any:
    """Fit on train rows only; supports bare classifier or Pipeline."""
    x = X_tr.values
    if sample_weight is not None:
        if isinstance(estimator, Pipeline):
            estimator.fit(x, y_tr_e, clf__sample_weight=sample_weight)
        else:
            estimator.fit(x, y_tr_e, sample_weight=sample_weight)
    else:
        estimator.fit(x, y_tr_e)
    return estimator


def predict_values(estimator: Any, X: pd.DataFrame) -> np.ndarray[Any, Any]:
    return np.asarray(estimator.predict(X.values))


def load_model_bundle(path: Path | str) -> dict[str, Any]:
    """Load ``model_bundle.joblib`` written by ``run_train``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    bundle = joblib.load(p)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict bundle, got {type(bundle).__name__}")
    return bundle


def validate_bundle_preprocess(
    bundle: dict[str, Any],
    cfg: ExperimentConfig,
) -> None:
    """Raise if YAML ``preprocess.scaler`` disagrees with the saved bundle."""
    stored = bundle.get("preprocess_scaler", "none")
    expected = cfg.preprocess.scaler
    if stored != expected:
        raise ValueError(
            f"Bundle preprocess_scaler={stored!r} but config has {expected!r}",
        )


def predict_from_bundle(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> np.ndarray[Any, Any]:
    """Apply saved pipeline/classifier to ``X`` (columns must match training)."""
    cols = feature_columns if feature_columns is not None else bundle["feature_columns"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Feature matrix missing columns: {missing}")
    extra = [c for c in X.columns if c not in cols]
    if extra:
        X = X[list(cols)]
    elif list(X.columns) != list(cols):
        X = X[list(cols)]
    return predict_values(bundle["estimator"], X)


def train_scaler_mean(estimator: Any) -> np.ndarray[Any, Any] | None:
    """Expose fitted scaler center for tests (StandardScaler.mean_)."""
    if not isinstance(estimator, Pipeline):
        return None
    scaler = estimator.named_steps.get("scaler")
    if scaler is None:
        return None
    center = getattr(scaler, "center_", None)
    if center is not None:
        return np.asarray(center)
    return np.asarray(getattr(scaler, "mean_", []))
