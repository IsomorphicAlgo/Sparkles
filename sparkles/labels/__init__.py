"""Triple-barrier labeling (Iteration 4)."""

from sparkles.labels.triple_barrier import (
    build_triple_barrier_labels,
    labeled_parquet_path,
    run_label,
)
from sparkles.labels.types import BarrierOutcome

__all__ = [
    "BarrierOutcome",
    "build_triple_barrier_labels",
    "labeled_parquet_path",
    "run_label",
]
