"""Join labeled entries with OHLCV for leakage-safe feature rows (Iteration 6).

Features use only information available **at the entry bar** (same timestamp as
``entry_time``): label-side barrier geometry and vol, plus that bar's OHLCV.
Trailing-window groups (Phase G1) read the full 1m series up to each entry bar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparkles.config.schema import ExperimentConfig, FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.intraday import max_warmup_bars as g1_warmup_bars
from sparkles.features.market_context import g3_warmup_bars
from sparkles.features.market_data import load_market_context_frames
from sparkles.features.registry import assemble_feature_columns
from sparkles.features.order_flow import g4c_warmup_bars
from sparkles.features.session import g2_warmup_bars
from sparkles.features.technical import g4a_warmup_bars
from sparkles.features.time import entry_session_dates
from sparkles.features.volatility import ensure_exchange_tz_index

logger = logging.getLogger(__name__)


def feature_warmup_bars(fc: FeatureConfig) -> int:
    return max(
        g1_warmup_bars(fc),
        g2_warmup_bars(fc),
        g3_warmup_bars(fc),
        g4a_warmup_bars(fc),
        g4c_warmup_bars(fc),
    )


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
    if fc.intraday_range_pct or fc.range_vol_multi or fc.vwap_distance:
        need.update({"high", "low"})
    if fc.bar_microstructure or fc.order_flow_proxies:
        need.update({"open", "high", "low"})
    if fc.log1p_volume or fc.volume_context or fc.vwap_distance or fc.order_flow_proxies:
        need.add("volume")
    if fc.returns_multi_horizon or fc.realized_vol_multi or fc.technical_indicators:
        need.add("close")
    return need


def _needs_full_ohlcv_history(fc: FeatureConfig) -> bool:
    return bool(
        fc.returns_multi_horizon
        or fc.realized_vol_multi
        or fc.range_vol_multi
        or fc.session_time
        or fc.volume_context
        or fc.vwap_distance
        or fc.market_context
        or fc.technical_indicators
        or fc.order_flow_proxies
    )


def build_feature_matrix(
    labels: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` aligned to labeled rows with matching OHLCV index.

    Drops label rows whose ``entry_time`` is missing from ``ohlcv``.
    Drops rows with NaN in any feature column (warm-up for trailing windows).
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

    positions = pd.Series(
        ohlcv.index.get_indexer(labels.index),
        index=labels.index,
        dtype=np.int64,
    )
    bad = positions.to_numpy(dtype=np.int64, copy=False) < 0
    if bool(bad.all()):
        raise ValueError("No label timestamps match ohlcv index")
    if bool(bad.any()):
        keep = ~bad
        labels = labels.loc[keep]
        positions = positions.loc[keep]

    aligned = ohlcv.reindex(labels.index)
    if aligned["close"].isna().any():
        ok = aligned["close"].notna()
        labels = labels.loc[ok]
        aligned = aligned.loc[ok]
        positions = positions.loc[ok]

    warmup = feature_warmup_bars(fc) if _needs_full_ohlcv_history(fc) else 0
    if warmup > 0:
        warm_ok = positions >= warmup
        dropped_warmup = int((~warm_ok).sum())
        if dropped_warmup:
            logger.info(
                "Dropping %s label rows before trailing-window warm-up (%s bars)",
                dropped_warmup,
                warmup,
            )
        labels = labels.loc[warm_ok]
        aligned = aligned.loc[warm_ok]
        positions = positions.loc[warm_ok]

    market_spy_1m: pd.DataFrame | None = None
    market_vix_1d: pd.DataFrame | None = None
    if fc.market_context:
        market_spy_1m, market_vix_1d = load_market_context_frames(
            cfg,
            base_dir=base_dir,
        )
        spy_ix = ensure_exchange_tz_index(market_spy_1m.index, cfg.exchange_timezone)
        entry_ix = ensure_exchange_tz_index(labels.index, cfg.exchange_timezone)
        spy_pos = pd.Series(
            spy_ix.get_indexer(entry_ix),
            index=labels.index,
            dtype=np.int64,
        )
        spy_warm = fc.market_spy_return_bars
        spy_ok = spy_pos >= spy_warm
        dropped_spy = int((~spy_ok).sum())
        if dropped_spy:
            logger.info(
                "Dropping %s label rows before SPY trailing-window warm-up (%s bars)",
                dropped_spy,
                spy_warm,
            )
        labels = labels.loc[spy_ok]
        aligned = aligned.loc[spy_ok]
        positions = positions.loc[spy_ok]

    ctx = EntryFeatureContext(
        labels=labels,
        aligned_ohlcv=aligned,
        entry_close=labels["entry_close"],
        full_ohlcv=ohlcv,
        entry_positions=positions,
        feature_config=fc,
        exchange_timezone=cfg.exchange_timezone,
        market_spy_1m=market_spy_1m,
        market_vix_1d=market_vix_1d,
    )
    X = assemble_feature_columns(ctx, fc)
    y = labels["barrier_outcome"].astype(str).copy()

    nan_rows = X.isna().any(axis=1)
    if bool(nan_rows.any()):
        n_nan = int(nan_rows.sum())
        logger.info(
            "Dropping %s label rows with NaN feature values after assembly",
            n_nan,
        )
        keep = ~nan_rows
        X = X.loc[keep]
        y = y.loc[keep]

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
