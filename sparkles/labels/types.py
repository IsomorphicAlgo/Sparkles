"""Label outcome types (Iteration 4)."""

from __future__ import annotations

from enum import Enum


class BarrierOutcome(str, Enum):
    """Which barrier was touched first (long, pessimistic same-bar tie-break)."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    VERTICAL = "vertical"
    END_OF_DATA = "end_of_data"
