"""Load and validate experiment YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from sparkles.config.schema import ExperimentConfig


def deep_merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto a copy of ``base`` (nested dicts only)."""
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge_mappings(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Parse experiment YAML into an ExperimentConfig.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if YAML is empty or not a mapping.
        pydantic.ValidationError: if YAML does not match the schema.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty YAML: {p}")
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)


def load_experiment_config_merged(
    base_path: Path | str,
    overlay_path: Path | str,
) -> ExperimentConfig:
    """Load ``base_path`` YAML and deep-merge ``overlay_path`` on top."""
    base_p = Path(base_path)
    overlay_p = Path(overlay_path)
    if not base_p.is_file():
        raise FileNotFoundError(f"Base config not found: {base_p}")
    if not overlay_p.is_file():
        raise FileNotFoundError(f"Overlay config not found: {overlay_p}")
    base_raw = yaml.safe_load(base_p.read_text(encoding="utf-8"))
    overlay_raw = yaml.safe_load(overlay_p.read_text(encoding="utf-8"))
    if base_raw is None or not isinstance(base_raw, dict):
        raise ValueError(f"Base YAML must be a mapping: {base_p}")
    if overlay_raw is None:
        overlay_raw = {}
    if not isinstance(overlay_raw, dict):
        raise ValueError(f"Overlay YAML must be a mapping: {overlay_p}")
    merged = deep_merge_mappings(base_raw, overlay_raw)
    return ExperimentConfig.model_validate(merged)
