"""Feature group registry: YAML ``features.*`` booleans → column builders."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import (
    EntryFeatureContext,
    build_intraday_range_pct,
    build_label_geometry,
    build_log1p_volume,
    build_log_entry_close,
)
from sparkles.features.intraday import (
    build_range_vol_multi,
    build_realized_vol_multi,
    build_returns_multi_horizon,
)
from sparkles.features.session import (
    build_session_day_of_week,
    build_session_time,
    build_volume_context,
    build_vwap_distance,
)
from sparkles.features.microstructure import build_bar_microstructure
from sparkles.features.market_context import build_market_context
from sparkles.features.order_flow import build_order_flow_proxies
from sparkles.features.technical import build_technical_indicators

# Stable column order for default all-on configs (matches pre–Phase B + G1/G2/G3 append).
_FEATURE_ORDER: Sequence[tuple[str, Callable[[EntryFeatureContext], pd.DataFrame]]] = (
    ("log_entry_close", build_log_entry_close),
    ("label_geometry", build_label_geometry),
    ("intraday_range_pct", build_intraday_range_pct),
    ("log1p_volume", build_log1p_volume),
    ("returns_multi_horizon", build_returns_multi_horizon),
    ("realized_vol_multi", build_realized_vol_multi),
    ("range_vol_multi", build_range_vol_multi),
    ("session_time", build_session_time),
    ("session_day_of_week", build_session_day_of_week),
    ("volume_context", build_volume_context),
    ("vwap_distance", build_vwap_distance),
    ("bar_microstructure", build_bar_microstructure),
    ("market_context", build_market_context),
    ("technical_indicators", build_technical_indicators),
    ("order_flow_proxies", build_order_flow_proxies),
)


def assemble_feature_columns(
    ctx: EntryFeatureContext,
    fc: FeatureConfig,
) -> pd.DataFrame:
    """Concatenate enabled builder outputs in canonical order."""
    parts: list[pd.DataFrame] = []
    for field, fn in _FEATURE_ORDER:
        if bool(getattr(fc, field)):
            parts.append(fn(ctx))
    if not parts:
        raise ValueError(
            "features config has no enabled groups; enable at least one of: "
            + ", ".join(f for f, _ in _FEATURE_ORDER),
        )
    return pd.concat(parts, axis=1)
