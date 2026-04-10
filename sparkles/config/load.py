"""Load and validate experiment YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from sparkles.config.schema import ExperimentConfig


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
