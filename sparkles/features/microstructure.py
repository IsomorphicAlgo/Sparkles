"""Entry-bar OHLC microstructure proxies (ML expansion Phase G3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparkles.features.builders import EntryFeatureContext


def build_bar_microstructure(ctx: EntryFeatureContext) -> pd.DataFrame:
    aligned = ctx.aligned_ohlcv
    hi = aligned["high"].astype(np.float64)
    lo = aligned["low"].astype(np.float64)
    op = aligned["open"].astype(np.float64)
    close = ctx.entry_close.astype(np.float64)
    span = (hi - lo).clip(lower=1e-12)
    close_loc = (close - lo) / span
    body_pct = (close - op) / close.clip(lower=1e-12)
    return pd.DataFrame(
        {
            "close_loc_value": close_loc,
            "bar_body_pct": body_pct,
        },
        index=close.index,
    )
