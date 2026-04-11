"""Estimator factory (Phase C)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from sparkles.config.schema import ExperimentConfig, ModelConfig
from sparkles.models import estimators


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_build_logistic_matches_yaml_params() -> None:
    cfg = _cfg(
        model=ModelConfig(
            type="logistic_regression",
            logistic_c=0.5,
            max_iter=500,
            random_seed=7,
            solver="saga",
            tol=1e-3,
        ),
    )
    est = estimators.build_estimator(cfg, logistic_class_weight="balanced")
    assert est.C == 0.5
    assert est.max_iter == 500
    assert est.random_state == 7
    assert est.solver == "saga"
    assert est.tol == 1e-3
    assert est.class_weight == "balanced"


def test_xgboost_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> type:
        raise ImportError(estimators._XGB_INSTALL_HINT)

    monkeypatch.setattr(estimators, "_xgboost_classifier_type", boom)
    cfg = _cfg(model=ModelConfig(type="xgboost_classifier"))
    with pytest.raises(ImportError, match=r"pip install.*\[ml\]"):
        estimators.build_estimator(cfg, logistic_class_weight=None)


def test_xgboost_fit_sample_weight_balanced() -> None:
    le = LabelEncoder()
    y = le.fit_transform(["a", "a", "a", "b"])
    mc = ModelConfig(type="xgboost_classifier", class_weight="balanced")
    sw = estimators.xgboost_fit_sample_weight(mc, le, y)
    assert sw is not None
    assert sw.shape == (4,)


def test_xgboost_classifier_smoke_fit() -> None:
    pytest.importorskip("xgboost")
    cfg = _cfg(
        model=ModelConfig(
            type="xgboost_classifier",
            xgb_n_estimators=5,
            xgb_max_depth=2,
            xgb_learning_rate=0.3,
        ),
    )
    est = estimators.build_estimator(cfg, logistic_class_weight=None)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    est.fit(X, y)
    assert len(est.predict(X)) == 4
