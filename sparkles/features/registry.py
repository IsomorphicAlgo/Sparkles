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
    build_session_time,
    build_volume_context,
    build_vwap_distance,
)

# Stable column order for default all-on configs (matches pre–Phase B + G1/G2 append).
_FEATURE_ORDER: Sequence[tuple[str, Callable[[EntryFeatureContext], pd.DataFrame]]] = (
    ("log_entry_close", build_log_entry_close),
    ("label_geometry", build_label_geometry),
    ("intraday_range_pct", build_intraday_range_pct),
    ("log1p_volume", build_log1p_volume),
    ("returns_multi_horizon", build_returns_multi_horizon),
    ("realized_vol_multi", build_realized_vol_multi),
    ("range_vol_multi", build_range_vol_multi),
    ("session_time", build_session_time),
    ("volume_context", build_volume_context),
    ("vwap_distance", build_vwap_distance),
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
