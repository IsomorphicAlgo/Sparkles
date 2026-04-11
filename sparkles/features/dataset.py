"""Join labeled entries with OHLCV for leakage-safe feature rows (Iteration 6).

Features use only information available **at the entry bar** (same timestamp as
``entry_time``): label-side barrier geometry and vol, plus that bar's OHLCV.
"""

from __future__ import annotations

import pandas as pd

from sparkles.config.schema import ExperimentConfig, FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.registry import assemble_feature_columns
from sparkles.features.volatility import ensure_exchange_tz_index


def entry_session_dates(
    index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> pd.Series:
    """US session calendar date per bar (normalized midnight in exchange TZ)."""
    ix = ensure_exchange_tz_index(pd.DatetimeIndex(index), exchange_timezone)
    norm = pd.DatetimeIndex(ix).normalize()
    return pd.Series(norm.date, index=index, dtype=object)


def _required_label_columns(fc: FeatureConfig) -> set[str]:
    need: set[str] = {"barrier_outcome"}
    if fc.log_entry_close or fc.intraday_range_pct:
        need.add("entry_close")
    if fc.label_geometry:
        need.update(
            {
                "sigma_ann_at_entry",
                "vol_scale_ratio",
                "tp_move_effective",
                "sl_move",
            },
        )
    return need


def _required_ohlcv_columns(fc: FeatureConfig) -> set[str]:
    need: set[str] = {"close"}
    if fc.intraday_range_pct:
        need.update({"high", "low"})
    return need


def build_feature_matrix(
    labels: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` aligned to labeled rows with matching OHLCV index.

    Drops label rows whose ``entry_time`` is missing from ``ohlcv``.
    """
    if labels.empty:
        raise ValueError("labels DataFrame is empty")
    fc = cfg.features
    need_l = _required_label_columns(fc)
    miss = need_l - set(labels.columns)
    if miss:
        raise KeyError(f"labels missing columns: {sorted(miss)}")

    need_o = _required_ohlcv_columns(fc)
    miss_o = need_o - set(ohlcv.columns)
    if miss_o:
        raise KeyError(f"ohlcv missing columns for enabled features: {sorted(miss_o)}")

    aligned = ohlcv.reindex(labels.index)
    bad = aligned["close"].isna()
    if bool(bad.all()):
        raise ValueError("No label timestamps match ohlcv index")
    if bool(bad.any()):
        labels = labels.loc[~bad]
        aligned = aligned.loc[~bad]

    ctx = EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=aligned,
        entry_close=labels["entry_close"],
    )
    X = assemble_feature_columns(ctx, fc)
    y = labels["barrier_outcome"].astype(str).copy()
    return X, y


def train_val_masks_by_session_date(
    index: pd.DatetimeIndex | pd.Index,
    cfg: ExperimentConfig,
) -> tuple[pd.Series, pd.Series]:
    """Boolean masks over ``index`` for train and val from config session dates."""
    if (
        cfg.train_start is None
        or cfg.train_end is None
        or cfg.val_start is None
        or cfg.val_end is None
    ):
        raise ValueError(
            "train_start, train_end, val_start, val_end must be set in the "
            "experiment YAML to run training",
        )
    d = entry_session_dates(index, cfg.exchange_timezone)
    ts0, ts1 = cfg.train_start, cfg.train_end
    vs0, vs1 = cfg.val_start, cfg.val_end
    train_m = (d >= ts0) & (d <= ts1)
    val_m = (d >= vs0) & (d <= vs1)
    return train_m, val_m
