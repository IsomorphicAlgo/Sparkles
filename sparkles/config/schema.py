"""Pydantic models for experiment YAML.

If you add new YAML keys, update this module and configs/experiments/*.yaml together.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Estimator settings (used from Iteration 6 onward)."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="logistic_regression",
        description="Estimator name, e.g. logistic_regression, xgb_classifier",
    )
    random_seed: int = Field(default=42, ge=0)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: str = "data/cache"
    artifacts_dir: str = "artifacts"


class ExperimentConfig(BaseModel):
    """Single experiment definition loaded from one YAML file."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker, e.g. RKLB")
    exchange_timezone: str = "America/New_York"

    cache_ttl_hours: int = Field(default=24, ge=1)
    data_start: date
    data_end: date

    profit_barrier_base: float = Field(default=0.15, gt=0, le=1.0)
    stop_loss_base: float = Field(default=0.05, gt=0, le=1.0)
    min_profit_per_trade_pct: float = Field(
        default=0.02,
        ge=0,
        le=1.0,
        description="Floor on TP move (fraction, e.g. 0.02 = 2%) after vol scaling",
    )
    vol_lookback_trading_days: int = Field(default=20, ge=2)
    vertical_max_trading_days: int = Field(
        default=10,
        ge=1,
        description="Triple-barrier vertical horizon in trading days",
    )
    vol_ref_method: Literal["median", "mean"] = "median"

    max_day_trades: int = Field(default=3, ge=1, le=3)
    rolling_business_days: int = Field(default=5, ge=1)

    train_start: date | None = None
    train_end: date | None = None
    val_start: date | None = None
    val_end: date | None = None

    model: ModelConfig = Field(default_factory=lambda: ModelConfig())
    paths: PathsConfig = Field(default_factory=lambda: PathsConfig())

    @model_validator(mode="after")
    def check_date_ranges(self) -> ExperimentConfig:
        if self.data_end < self.data_start:
            raise ValueError("data_end must be on or after data_start")
        if self.train_start is not None and self.train_end is not None:
            if self.train_end < self.train_start:
                raise ValueError("train_end must be on or after train_start")
        if self.val_start is not None and self.val_end is not None:
            if self.val_end < self.val_start:
                raise ValueError("val_end must be on or after val_start")
        return self
