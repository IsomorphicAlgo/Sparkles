"""Label outcome types (Iteration 4)."""

from __future__ import annotations

from enum import Enum


class BarrierOutcome(str, Enum):
    """Which barrier was touched first (long position convention)."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    VERTICAL = "vertical"
    # Extend in Iteration 4 as needed (e.g. neutral / no trade)
