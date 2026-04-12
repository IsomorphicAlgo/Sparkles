"""LiveIngestConfig (Phase 2 Plan A) validation."""

from __future__ import annotations

import pytest

from sparkles.config.schema import FeatureConfig, LiveIngestConfig


def test_live_ingest_defaults_batch_only() -> None:
    c = LiveIngestConfig()
    assert c.enabled is False
    assert c.poll_interval_seconds == 120.0
    assert c.merge_strategy == "separate_recent_parquet"
    assert c.session_start_local is None


def test_live_ingest_rejects_poll_below_floor() -> None:
    with pytest.raises(ValueError):
        LiveIngestConfig(poll_interval_seconds=30.0)


def test_live_ingest_session_both_required() -> None:
    with pytest.raises(ValueError, match="both"):
        LiveIngestConfig(session_start_local="09:30")


def test_live_ingest_session_end_after_start() -> None:
    with pytest.raises(ValueError, match="after"):
        LiveIngestConfig(
            session_start_local="16:00",
            session_end_local="09:30",
        )


def test_live_ingest_valid_window() -> None:
    c = LiveIngestConfig(
        session_start_local="04:00",
        session_end_local="20:00",
    )
    assert c.session_start_local == "04:00"


def test_live_ingest_invalid_hhmm() -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        LiveIngestConfig(
            session_start_local="25:00",
            session_end_local="26:00",
        )


def test_feature_config_still_validates() -> None:
    """Regression: nested configs unchanged."""
    FeatureConfig()
    with pytest.raises(ValueError, match="enable at least one"):
        FeatureConfig(
            log_entry_close=False,
            label_geometry=False,
            intraday_range_pct=False,
            log1p_volume=False,
        )
