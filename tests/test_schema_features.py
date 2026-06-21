"""FeatureConfig validation."""

from __future__ import annotations

import pytest

from sparkles.config.schema import FeatureConfig


def test_feature_config_rejects_all_disabled() -> None:
    with pytest.raises(ValueError, match="enable at least one"):
        FeatureConfig(
            log_entry_close=False,
            label_geometry=False,
            intraday_range_pct=False,
            log1p_volume=False,
            returns_multi_horizon=False,
            realized_vol_multi=False,
            range_vol_multi=False,
            session_time=False,
            volume_context=False,
            vwap_distance=False,
            bar_microstructure=False,
            market_context=False,
        )
