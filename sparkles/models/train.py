"""Training entrypoint: split, fit, save artifact (Iteration 6).

Edit this file for day-to-day model experiments. Stable hyperparameters also
live under `model:` in configs/experiments/*.yaml.

Typical flow (once implemented): load labeled parquet → build X, y →
build_estimator() → fit → save via registry.
"""

from __future__ import annotations

from typing import Any

# Quick knobs while iterating in Python (mirror YAML when stable)
DEFAULT_TRAIN_KWARGS: dict[str, Any] = {}


def build_estimator(config_type: str) -> Any:
    """Return an unfitted estimator (Iteration 6)."""
    raise NotImplementedError(
        f"sparkles.models.train.build_estimator({config_type!r}) — Iteration 6",
    )


def placeholder() -> None:
    """Implemented in Iteration 6."""
    raise NotImplementedError("sparkles.models.train — Iteration 6")
