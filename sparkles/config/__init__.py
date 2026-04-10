"""Experiment configuration (YAML + Pydantic)."""

from sparkles.config.load import load_experiment_config
from sparkles.config.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "load_experiment_config"]
