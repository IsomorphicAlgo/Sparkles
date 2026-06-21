"""AFML-style sample weights for overlapping triple-barrier labels (Phase I4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sparkles.backtest.pnl import lookup_ohlcv_position
from sparkles.config.schema import ExperimentConfig
from sparkles.features.volatility import ensure_exchange_tz_index
from sparkles.models.estimators import resolve_logistic_class_weight


def entry_bar_positions(
    entry_times: pd.DatetimeIndex | pd.Index,
    ohlcv_index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> np.ndarray[Any, Any]:
    """Integer OHLCV positions for each labeled entry (NaN if missing)."""
    ix = ensure_exchange_tz_index(ohlcv_index, exchange_timezone)
    positions = np.empty(len(entry_times), dtype=np.float64)
    for i, ts in enumerate(pd.DatetimeIndex(entry_times)):
        pos = lookup_ohlcv_position(ix, pd.Timestamp(ts))
        positions[i] = np.nan if pos is None else float(pos)
    return positions


def uniqueness_weights(
    entry_positions: np.ndarray[Any, Any],
    bars_forward: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Average inverse concurrency over each label's forward window (AFML-style).

    Each label spans bars ``(entry_pos, entry_pos + bars_forward]`` inclusive of
    forward steps. Weight for label *i* is the mean of ``1 / c(t)`` over that span,
    where *c(t)* counts how many label windows cover bar *t*.
    """
    n = len(entry_positions)
    out = np.ones(n, dtype=np.float64)
    if n == 0:
        return out

    valid = np.isfinite(entry_positions) & np.isfinite(bars_forward)
    if not np.any(valid):
        return out

    pos = entry_positions[valid].astype(np.int64)
    fwd = bars_forward[valid].astype(np.int64)
    if np.any(fwd < 1):
        raise ValueError("bars_forward must be >= 1 for uniqueness weights")

    max_end = int(np.max(pos + fwd))
    concurrency = np.zeros(max_end + 1, dtype=np.int64)
    for p, f in zip(pos, fwd, strict=True):
        concurrency[p + 1 : p + f + 1] += 1

    w_valid = np.empty(len(pos), dtype=np.float64)
    for i, (p, f) in enumerate(zip(pos, fwd, strict=True)):
        span = concurrency[p + 1 : p + f + 1]
        if span.size == 0 or np.any(span <= 0):
            w_valid[i] = 1.0
        else:
            w_valid[i] = float(np.mean(1.0 / span.astype(np.float64)))

    out[np.where(valid)[0]] = w_valid
    return out


def class_weight_vector(
    cfg: ExperimentConfig,
    le: LabelEncoder,
    y_tr_e: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any] | None:
    """Per-row class weights from ``model.class_weight`` (None if unset)."""
    if cfg.model.class_weight is None:
        return None
    if cfg.model.type == "logistic_regression":
        resolved = resolve_logistic_class_weight(cfg.model, le)
        if resolved is None or resolved == "balanced":
            return np.asarray(
                compute_sample_weight(resolved, y_tr_e),
                dtype=np.float64,
            )
        return np.asarray(compute_sample_weight(resolved, y_tr_e), dtype=np.float64)
    if cfg.model.type == "xgboost_classifier":
        if cfg.model.class_weight == "balanced":
            return np.asarray(
                compute_sample_weight("balanced", y_tr_e),
                dtype=np.float64,
            )
        resolved = resolve_logistic_class_weight(cfg.model, le)
        assert isinstance(resolved, dict)
        return np.asarray(compute_sample_weight(resolved, y_tr_e), dtype=np.float64)
    return None


def resolve_fit_sample_weights(
    cfg: ExperimentConfig,
    le: LabelEncoder,
    y_tr_e: np.ndarray[Any, Any],
    entry_times: pd.DatetimeIndex | pd.Index,
    labels: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> tuple[np.ndarray[Any, Any] | None, dict[str, float | str | None]]:
    """Build optional ``sample_weight`` vector for ``fit`` and a small summary dict."""
    method = cfg.train.sample_weight_method
    summary: dict[str, float | str | None] = {
        "sample_weight_method": method,
        "sample_weight_mean": None,
        "sample_weight_min": None,
    }
    if method == "none":
        cw = class_weight_vector(cfg, le, y_tr_e)
        if cw is None:
            return None, summary
        summary["sample_weight_mean"] = float(np.mean(cw))
        summary["sample_weight_min"] = float(np.min(cw))
        return cw, summary

    if method != "uniqueness":
        raise ValueError(f"Unsupported train.sample_weight_method: {method!r}")

    aligned = labels.reindex(pd.DatetimeIndex(entry_times))
    if aligned["bars_forward"].isna().any():
        raise ValueError("Missing bars_forward for some train rows")

    positions = entry_bar_positions(
        entry_times,
        ohlcv.index,
        cfg.exchange_timezone,
    )
    if np.isnan(positions).any():
        raise ValueError("Some train entry times missing from OHLCV index")

    uniq = uniqueness_weights(
        positions,
        aligned["bars_forward"].to_numpy(dtype=np.float64),
    )
    cw = class_weight_vector(cfg, le, y_tr_e)
    if cw is None:
        combined = uniq
    else:
        combined = uniq * cw

    summary["sample_weight_mean"] = float(np.mean(combined))
    summary["sample_weight_min"] = float(np.min(combined))
    return combined, summary
