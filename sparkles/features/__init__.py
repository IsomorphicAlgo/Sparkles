"""Feature builders (Iteration 3+)."""

from sparkles.features.dataset import (
    build_feature_matrix,
    entry_session_dates,
    train_val_masks_by_session_date,
)
from sparkles.features.volatility import (
    TRADING_DAYS_PER_YEAR,
    add_volatility_columns,
    add_volatility_from_config,
    align_volatility_to_1m_index,
    daily_last_close,
    daily_log_returns,
    ensure_exchange_tz_index,
    rolling_volatility_daily_returns_no_lookahead,
)

__all__ = [
    "build_feature_matrix",
    "entry_session_dates",
    "train_val_masks_by_session_date",
    "TRADING_DAYS_PER_YEAR",
    "add_volatility_columns",
    "add_volatility_from_config",
    "align_volatility_to_1m_index",
    "daily_last_close",
    "daily_log_returns",
    "ensure_exchange_tz_index",
    "rolling_volatility_daily_returns_no_lookahead",
]
