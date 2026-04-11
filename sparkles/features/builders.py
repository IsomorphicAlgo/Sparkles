"""Entry-time feature column builders (ML expansion Phase B).

Each builder returns a small ``DataFrame`` aligned to ``labels.index`` using only
information available at the entry bar (see ``EntryFeatureContext``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntryFeatureContext:
    """Aligned label rows and OHLCV at the entry timestamp."""

    labels: pd.DataFrame
    aligned_ohlcv: pd.DataFrame
    entry_close: pd.Series


def build_log_entry_close(ctx: EntryFeatureContext) -> pd.DataFrame:
    c = ctx.entry_close.astype(np.float64)
    return pd.DataFrame(
        {"log_entry_close": np.log(c.clip(lower=1e-12))},
        index=c.index,
    )


def build_label_geometry(ctx: EntryFeatureContext) -> pd.DataFrame:
    lab = ctx.labels
    return pd.DataFrame(
        {
            "sigma_ann_at_entry": lab["sigma_ann_at_entry"].astype(np.float64),
            "vol_scale_ratio": lab["vol_scale_ratio"].astype(np.float64),
            "tp_move_effective": lab["tp_move_effective"].astype(np.float64),
            "sl_move": lab["sl_move"].astype(np.float64),
        },
        index=lab.index,
    )


def build_intraday_range_pct(ctx: EntryFeatureContext) -> pd.DataFrame:
    aligned = ctx.aligned_ohlcv
    close = ctx.entry_close.astype(np.float64)
    hi = aligned["high"].astype(np.float64)
    lo = aligned["low"].astype(np.float64)
    return pd.DataFrame(
        {"intraday_range_pct": (hi - lo) / close.clip(lower=1e-12)},
        index=close.index,
    )


def build_log1p_volume(ctx: EntryFeatureContext) -> pd.DataFrame:
    aligned = ctx.aligned_ohlcv
    if "volume" in aligned.columns:
        vol = aligned["volume"].astype(np.float64)
    else:
        vol = pd.Series(0.0, index=aligned.index, dtype=np.float64)
    return pd.DataFrame(
        {"log1p_volume": np.log1p(np.maximum(vol, 0.0))},
        index=aligned.index,
    )
