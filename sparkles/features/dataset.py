"""Join labeled entries with OHLCV for leakage-safe feature rows (Iteration 6).

Features use only information available **at the entry bar** (same timestamp as
``entry_time``): label-side barrier geometry and vol, plus that bar's OHLCV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sparkles.features.volatility import ensure_exchange_tz_index

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig


def entry_session_dates(
    index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> pd.Series:
    """US session calendar date per bar (normalized midnight in exchange TZ)."""
    ix = ensure_exchange_tz_index(pd.DatetimeIndex(index), exchange_timezone)
    norm = pd.DatetimeIndex(ix).normalize()
    return pd.Series(norm.date, index=index, dtype=object)


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
    need_l = {
        "entry_close",
        "barrier_outcome",
        "sigma_ann_at_entry",
        "vol_scale_ratio",
        "tp_move_effective",
        "sl_move",
    }
    miss = need_l - set(labels.columns)
    if miss:
        raise KeyError(f"labels missing columns: {sorted(miss)}")

    aligned = ohlcv.reindex(labels.index)
    bad = aligned["close"].isna()
    if bool(bad.all()):
        raise ValueError("No label timestamps match ohlcv index")
    if bool(bad.any()):
        labels = labels.loc[~bad]
        aligned = aligned.loc[~bad]

    close = labels["entry_close"].astype(np.float64)
    hi = aligned["high"].astype(np.float64)
    lo = aligned["low"].astype(np.float64)
    if "volume" in aligned.columns:
        vol = aligned["volume"].astype(np.float64)
    else:
        vol = pd.Series(0.0, index=aligned.index, dtype=np.float64)

    X = pd.DataFrame(
        {
            "log_entry_close": np.log(close.clip(lower=1e-12)),
            "sigma_ann_at_entry": labels["sigma_ann_at_entry"].astype(np.float64),
            "vol_scale_ratio": labels["vol_scale_ratio"].astype(np.float64),
            "tp_move_effective": labels["tp_move_effective"].astype(np.float64),
            "sl_move": labels["sl_move"].astype(np.float64),
            "intraday_range_pct": (hi - lo) / close.clip(lower=1e-12),
            "log1p_volume": np.log1p(np.maximum(vol, 0.0)),
        },
        index=labels.index,
    )
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
