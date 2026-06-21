"""Preprocessing pipeline tests (Phase D — no val leakage)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sparkles.config.schema import ExperimentConfig, PreprocessConfig
from sparkles.models import preprocess
from sparkles.models.train import run_train


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
        "train_start": date(2024, 6, 3),
        "train_end": date(2024, 6, 4),
        "val_start": date(2024, 6, 5),
        "val_end": date(2024, 6, 7),
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_build_training_estimator_none_returns_classifier() -> None:
    cfg = _cfg(preprocess=PreprocessConfig(scaler="none"))
    clf = LogisticRegression(max_iter=200)
    est = preprocess.build_training_estimator(cfg, clf)
    assert not isinstance(est, Pipeline)
    assert est is clf


def test_build_training_estimator_standard_returns_pipeline() -> None:
    cfg = _cfg(preprocess=PreprocessConfig(scaler="standard"))
    clf = LogisticRegression(max_iter=200)
    est = preprocess.build_training_estimator(cfg, clf)
    assert isinstance(est, Pipeline)
    assert "scaler" in est.named_steps
    assert "clf" in est.named_steps


def test_scaler_fit_on_train_only_not_val_distribution() -> None:
    """Val rows with a very different mean must not shrink scaler center toward val."""
    cfg = _cfg(preprocess=PreprocessConfig(scaler="standard"))
    clf = LogisticRegression(max_iter=500, random_state=0)
    pipe = preprocess.build_training_estimator(cfg, clf)

    x_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_tr = np.array([0, 0, 1])
    x_va = pd.DataFrame({"a": [100.0, 200.0]})

    preprocess.fit_training_estimator(pipe, x_tr, y_tr)
    center = preprocess.train_scaler_mean(pipe)
    assert center is not None
    assert center[0] == pytest.approx(2.0)

    va_scaled = pipe.named_steps["scaler"].transform(x_va.values)
    assert va_scaled[0, 0] > 10.0


def test_bundle_reload_predict_matches_direct(tmp_path: Path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "high": [101.0, 111.0, 121.0, 131.0, 141.0, 151.0, 161.0, 171.0],
            "low": [99.0, 109.0, 119.0, 129.0, 139.0, 149.0, 159.0, 169.0],
            "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": ohlcv["close"].values,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg(preprocess=PreprocessConfig(scaler="standard"))
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    bundle_path = out / "model_bundle.joblib"
    bundle = preprocess.load_model_bundle(bundle_path)
    assert bundle["preprocess_scaler"] == "standard"
    preprocess.validate_bundle_preprocess(bundle, cfg)

    wrong = _cfg(preprocess=PreprocessConfig(scaler="none"))
    with pytest.raises(ValueError, match="preprocess_scaler"):
        preprocess.validate_bundle_preprocess(bundle, wrong)

    from sparkles.features.dataset import build_feature_matrix

    x_all, _ = build_feature_matrix(labels, ohlcv, cfg)
    x_va_only = x_all.loc[idx[4:6]]
    direct = preprocess.predict_values(bundle["estimator"], x_va_only)
    via_helper = preprocess.predict_from_bundle(bundle, x_va_only)
    np.testing.assert_array_equal(direct, via_helper)


def test_run_train_records_preprocess_in_metrics(tmp_path: Path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": [
                "take_profit",
                "stop_loss",
                "vertical",
                "end_of_data",
                "take_profit",
                "stop_loss",
                "vertical",
                "take_profit",
            ],
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg(preprocess=PreprocessConfig(scaler="robust"))
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    import json

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["preprocess_scaler"] == "robust"
    assert isinstance(preprocess.load_model_bundle(out / "model_bundle.joblib")["estimator"], Pipeline)
