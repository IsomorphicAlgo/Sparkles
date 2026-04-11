"""Build per-row prediction tables for Parquet export next to ``metrics.json``."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sparkles.features.dataset import entry_session_dates


def _proba_column_name(class_label: str) -> str:
    safe = re.sub(r"[^\w]+", "_", str(class_label), flags=re.ASCII).strip("_")
    return f"proba_{safe}"


def predictions_frame(
    X: pd.DataFrame,
    y_true_str: pd.Series,
    y_enc: np.ndarray[Any, Any],
    pred_enc: np.ndarray[Any, Any],
    split: str,
    estimator: Any,
    le: LabelEncoder,
    exchange_timezone: str,
) -> pd.DataFrame:
    """One row per entry bar: time, session date, split, labels, optional proba."""
    entry_time = pd.DatetimeIndex(X.index)
    session = entry_session_dates(entry_time, exchange_timezone)
    y_pred_str = le.inverse_transform(pred_enc.astype(np.int64))

    out: dict[str, Any] = {
        "entry_time": entry_time,
        "session_date": session.values,
        "split": split,
        "y_true": y_true_str.astype(str).values,
        "y_pred": y_pred_str.astype(str),
    }

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X.values.astype(np.float64))
        for j, cls in enumerate(le.classes_):
            out[_proba_column_name(str(cls))] = proba[:, j].astype(np.float64)
        out["max_proba"] = proba.max(axis=1).astype(np.float64)

    return pd.DataFrame(out)
