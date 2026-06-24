"""Pydantic models for experiment YAML.

If you add new YAML keys, update this module and configs/experiments/*.yaml together.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

ModelKind = Literal["logistic_regression", "xgboost_classifier"]

LogisticSolver = Literal[
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
]


class ModelConfig(BaseModel):
    """Estimator settings (used from Iteration 6 onward)."""

    model_config = ConfigDict(extra="forbid")

    type: ModelKind = Field(
        default="logistic_regression",
        description=(
            "Classifier: logistic_regression (core) or "
            "xgboost_classifier ([ml] extra)"
        ),
    )
    random_seed: int = Field(default=42, ge=0)
    logistic_c: float = Field(
        default=1.0,
        gt=0,
        description="sklearn LogisticRegression inverse regularization strength C",
    )
    max_iter: int = Field(
        default=2000,
        ge=100,
        le=100_000,
        description="Max iterations for iterative solvers (logistic_regression)",
    )
    solver: LogisticSolver = Field(
        default="lbfgs",
        description="sklearn LogisticRegression solver",
    )
    tol: float = Field(
        default=1e-4,
        gt=0,
        le=1.0,
        description="sklearn LogisticRegression stopping tolerance",
    )
    class_weight: None | Literal["balanced"] | dict[str, float] = Field(
        default=None,
        description="None, 'balanced', or {barrier_outcome class name: weight}",
    )

    @field_validator("class_weight", mode="before")
    @classmethod
    def coerce_class_weight(cls, v: object) -> object:
        if v is None or v == "balanced":
            return v
        if isinstance(v, dict):
            out: dict[str, float] = {}
            for k, w in v.items():
                out[str(k)] = float(w)  # YAML may use numeric keys
            return out
        if isinstance(v, str):
            raise ValueError(
                "model.class_weight string must be 'balanced' or omit for None",
            )
        raise TypeError("model.class_weight must be null, 'balanced', or a mapping")

    # XGBoost when type == xgboost_classifier (optional pip install -e ".[ml]").
    xgb_n_estimators: int = Field(
        default=100,
        ge=1,
        le=50_000,
        description="XGBClassifier n_estimators",
    )
    xgb_max_depth: int = Field(
        default=6,
        ge=1,
        le=30,
        description="XGBClassifier max_depth",
    )
    xgb_learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="XGBClassifier learning_rate",
    )
    xgb_subsample: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="XGBClassifier subsample",
    )
    xgb_colsample_bytree: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="XGBClassifier colsample_bytree",
    )
    xgb_min_child_weight: float = Field(
        default=1.0,
        ge=0,
        description="XGBClassifier min_child_weight",
    )


class FeatureConfig(BaseModel):
    """Entry-time feature groups toggled from YAML (ML expansion Phase B).

    Each ``True`` flag includes one builder block in ``sparkles/features/registry.py``.
    """

    model_config = ConfigDict(extra="forbid")

    log_entry_close: bool = Field(
        default=True,
        description="log of entry_close (from labels)",
    )
    label_geometry: bool = Field(
        default=True,
        description="sigma_ann_at_entry, vol_scale_ratio, tp_move_effective, sl_move",
    )
    intraday_range_pct: bool = Field(
        default=True,
        description="(high-low)/entry_close on the entry bar from OHLCV",
    )
    log1p_volume: bool = Field(
        default=True,
        description="log1p(volume) on the entry bar",
    )
    returns_multi_horizon: bool = Field(
        default=False,
        description="Trailing log returns over returns_horizons_bars 1m bars",
    )
    returns_horizons_bars: list[int] = Field(
        default_factory=lambda: [5, 15, 30, 60],
        description="Lookback bar counts for returns_multi_horizon columns",
    )
    realized_vol_multi: bool = Field(
        default=False,
        description="Trailing 1m realized vol (std of log returns) per window",
    )
    realized_vol_windows_bars: list[int] = Field(
        default_factory=lambda: [30, 120],
        description="Window lengths for realized_vol_multi columns",
    )
    realized_vol_include_ratio: bool = Field(
        default=True,
        description="When >=2 windows, add short/long rv ratio column",
    )
    range_vol_multi: bool = Field(
        default=False,
        description="Parkinson range vol and optional normalized ATR",
    )
    range_vol_window_bars: int = Field(
        default=30,
        ge=2,
        description="Rolling window for range_vol_multi",
    )
    range_vol_include_atr_norm: bool = Field(
        default=True,
        description="Include atr_norm_{window}m when range_vol_multi is on",
    )
    session_time: bool = Field(
        default=False,
        description="minutes_since_open, minutes_to_close, sin_time, cos_time",
    )
    session_day_of_week: bool = Field(
        default=False,
        description="Phase G4b: sin_dow, cos_dow cyclical weekday (Mon=0 … Fri=4)",
    )
    volume_context: bool = Field(
        default=False,
        description="rel_volume, log_rel_volume vs trailing median volume",
    )
    volume_median_window_bars: int = Field(
        default=60,
        ge=2,
        description="Trailing window for volume_context median",
    )
    vwap_distance: bool = Field(
        default=False,
        description="(close - session VWAP) / session VWAP through entry bar",
    )
    bar_microstructure: bool = Field(
        default=False,
        description="close_loc_value, bar_body_pct on the entry bar",
    )
    market_context: bool = Field(
        default=False,
        description="spy_ret_{N}m and vix_chg_1d from context_ingest Parquet",
    )
    market_spy_return_bars: int = Field(
        default=15,
        ge=1,
        description="Trailing SPY 1m log-return lookback when market_context is on",
    )
    technical_indicators: bool = Field(
        default=False,
        description="Phase G4a: EMA distance, Wilder RSI, MACD on 1m close at entry",
    )
    ema_windows_bars: list[int] = Field(
        default_factory=lambda: [9, 21, 50],
        description="EMA lookback spans (bars) for ema_dist_{w}m columns",
    )
    rsi_window_bars: int = Field(
        default=14,
        ge=2,
        description="Wilder RSI lookback; column rsi_{N}m scaled to [0, 1]",
    )
    macd_fast_bars: int = Field(
        default=12,
        ge=2,
        description="MACD fast EMA span",
    )
    macd_slow_bars: int = Field(
        default=26,
        ge=3,
        description="MACD slow EMA span",
    )
    macd_signal_bars: int = Field(
        default=9,
        ge=2,
        description="MACD signal EMA span",
    )
    order_flow_proxies: bool = Field(
        default=False,
        description="Phase G4c: OHLCV spread/illiquidity proxies (no L2 data)",
    )
    roll_window_bars: int = Field(
        default=20,
        ge=2,
        description="Trailing window for Corwin–Schultz and Roll spread proxies",
    )
    amihud_window_bars: int = Field(
        default=20,
        ge=2,
        description="Trailing window for Amihud illiquidity proxy",
    )

    @field_validator("ema_windows_bars")
    @classmethod
    def _ema_windows_positive(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("features.ema_windows_bars must not be empty when used")
        if any(w < 2 for w in v):
            raise ValueError("features.ema_windows_bars: each value must be >= 2")
        return sorted(set(v))

    @model_validator(mode="after")
    def _macd_fast_slower_than_slow(self) -> FeatureConfig:
        if self.macd_fast_bars >= self.macd_slow_bars:
            raise ValueError(
                "features.macd_fast_bars must be < macd_slow_bars "
                f"(got {self.macd_fast_bars} >= {self.macd_slow_bars})",
            )
        return self

    @field_validator("returns_horizons_bars")
    @classmethod
    def _returns_horizons_positive(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("features.returns_horizons_bars must not be empty")
        if any(h < 1 for h in v):
            raise ValueError("features.returns_horizons_bars: each value must be >= 1")
        return sorted(set(v))

    @field_validator("realized_vol_windows_bars")
    @classmethod
    def _realized_vol_windows_valid(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("features.realized_vol_windows_bars must not be empty")
        if any(w < 2 for w in v):
            raise ValueError(
                "features.realized_vol_windows_bars: each value must be >= 2",
            )
        return sorted(set(v))

    @model_validator(mode="after")
    def at_least_one_group(self) -> FeatureConfig:
        if not any(
            (
                self.log_entry_close,
                self.label_geometry,
                self.intraday_range_pct,
                self.log1p_volume,
                self.returns_multi_horizon,
                self.realized_vol_multi,
                self.range_vol_multi,
                self.session_time,
                self.session_day_of_week,
                self.volume_context,
                self.vwap_distance,
                self.bar_microstructure,
                self.market_context,
                self.technical_indicators,
                self.order_flow_proxies,
            ),
        ):
            raise ValueError(
                "features: enable at least one feature group "
                "(log_entry_close, label_geometry, intraday_range_pct, log1p_volume, "
                "returns_multi_horizon, realized_vol_multi, range_vol_multi, "
                "session_time, volume_context, vwap_distance, bar_microstructure, "
                "market_context, technical_indicators, session_day_of_week, "
                "order_flow_proxies)",
            )
        return self


class ContextSymbolConfig(BaseModel):
    """One extra symbol to cache for market_context (Phase G3)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, description="Ticker, e.g. SPY or VIX")
    interval: Literal["1min", "1day"] = Field(
        default="1min",
        description="TwelveData interval; use 1day for VIX daily change feature",
    )
    twelvedata_exchange: str | None = Field(
        default=None,
        description="Optional exchange id for TwelveData",
    )


class ContextIngestConfig(BaseModel):
    """Optional context symbols downloaded with the same data_start/data_end as symbol."""

    model_config = ConfigDict(extra="forbid")

    symbols: list[ContextSymbolConfig] = Field(
        default_factory=list,
        description="Extra tickers cached by sparkles ingest (SPY 1min, VIX 1day, etc.)",
    )


class PreprocessConfig(BaseModel):
    """Feature scaling before the classifier (ML expansion Phase D)."""

    model_config = ConfigDict(extra="forbid")

    scaler: Literal["none", "standard", "robust"] = Field(
        default="none",
        description=(
            "Fit scaler on train rows only inside a sklearn Pipeline: "
            "none | standard (StandardScaler) | robust (RobustScaler)"
        ),
    )


class TrainConfig(BaseModel):
    """Training run options (ML expansion Phase A)."""

    model_config = ConfigDict(extra="forbid")

    min_train_rows: int = Field(
        default=1,
        ge=1,
        description="Minimum labeled train rows after the date split (before fit)",
    )
    min_val_rows: int = Field(
        default=1,
        ge=1,
        description="Minimum val rows after optional unseen-class filtering",
    )
    drop_val_unseen_classes: bool = Field(
        default=True,
        description="If true, drop val rows whose outcome was not seen in train",
    )
    experiment_name: str | None = Field(
        default=None,
        description="Optional short name appended to experiments.jsonl",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-text note stored in experiments.jsonl",
    )
    export_predictions: Literal["none", "val", "all"] = Field(
        default="val",
        description=(
            "Write predictions.parquet: val rows, train+val, or none (skip file)"
        ),
    )
    entry_threshold_take_profit: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Optional backtest entry rule: enter when proba_take_profit >= threshold "
            "(default backtest policy remains argmax when null)"
        ),
    )
    meta_label_primary_threshold: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Phase I3: primary proba_take_profit gate for meta-label train/compare "
            "(falls back to entry_threshold_take_profit, then 0.35)"
        ),
    )
    meta_label_act_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Phase I3: minimum meta-model act probability on gated primary signals",
    )
    sample_weight_method: Literal["none", "uniqueness"] = Field(
        default="none",
        description=(
            "Phase I4: per-row fit weights — 'uniqueness' down-weights overlapping "
            "triple-barrier labels (uses bars_forward); default none preserves prior runs"
        ),
    )


class JournalConfig(BaseModel):
    """Optional personal trade log for alignment with model predictions (CSV)."""

    model_config = ConfigDict(extra="forbid")

    csv_path: str | None = Field(
        default=None,
        description="Path to trades CSV (repo-relative or absolute); see DEVELOPER.md",
    )


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: str = "data/cache"
    artifacts_dir: str = "artifacts"


class LiveIngestConfig(BaseModel):
    """Phase 2 Plan A: interval / near-live refresh (behavior starts in A3+).

    Validated in Iteration A1; defaults keep Phase 1 batch-only workflows unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "If true, future refresh/loop commands may run; false = batch ingest only"
        ),
    )
    poll_interval_seconds: float = Field(
        default=120.0,
        ge=60.0,
        le=86_400.0,
        description=(
            "Minimum seconds between refresh API calls in the long-running loop (A4+); "
            "60s floor aligns with TwelveData free-tier credit discipline"
        ),
    )
    refresh_lookback_calendar_days: int = Field(
        default=2,
        ge=1,
        le=31,
        description=(
            "Calendar-day span each refresh request covers (caps API payload size)"
        ),
    )
    merge_strategy: Literal[
        "separate_recent_parquet",
        "merge_into_main_cache",
    ] = Field(
        default="separate_recent_parquet",
        description=(
            "separate_recent_parquet: write/update a sidecar *_recent.parquet; "
            "merge_into_main_cache: append into the main ingest Parquet (A3)"
        ),
    )
    include_extended_hours: bool = Field(
        default=False,
        description=(
            "If true, request extended-hours 1m where the API supports it; "
            "false = provider default (often regular hours)"
        ),
    )
    session_start_local: str | None = Field(
        default=None,
        description=(
            "Optional window start HH:MM 24h in exchange_timezone; null = no gate"
        ),
    )
    session_end_local: str | None = Field(
        default=None,
        description=(
            "Optional window end HH:MM 24h in exchange_timezone; null = no gate"
        ),
    )

    @field_validator("session_start_local", "session_end_local", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> object:
        if v == "":
            return None
        return v

    @field_validator("session_start_local", "session_end_local", mode="after")
    @classmethod
    def hhmm_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _HHMM.fullmatch(v):
            raise ValueError(
                "live_ingest session times must be HH:MM 24h (e.g. 04:00)",
            )
        return v

    @model_validator(mode="after")
    def session_both_or_neither(self) -> LiveIngestConfig:
        a, b = self.session_start_local, self.session_end_local
        if (a is None) ^ (b is None):
            raise ValueError(
                "live_ingest: set both session_start_local and session_end_local, "
                "or neither",
            )
        return self

    @model_validator(mode="after")
    def session_end_after_start_same_day(self) -> LiveIngestConfig:
        a, b = self.session_start_local, self.session_end_local
        if a is None or b is None:
            return self
        ha, ma = (int(x) for x in a.split(":"))
        hb, mb = (int(x) for x in b.split(":"))
        if (ha, ma) >= (hb, mb):
            raise ValueError(
                "live_ingest: session_end_local must be after session_start_local "
                "(same calendar day; overnight windows not in v1)",
            )
        return self


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
    barrier_vol_scale_min: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description="Clamp sigma/sigma_ref from below when scaling barriers",
    )
    barrier_vol_scale_max: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Clamp sigma/sigma_ref from above when scaling barriers",
    )
    label_entry_stride: int = Field(
        default=390,
        ge=1,
        description="Use every Nth bar as entry (390 ≈ one per 1m trading day)",
    )
    label_cache_suffix: str | None = Field(
        default=None,
        description=(
            "Optional tag in labeled Parquet filename when barrier presets differ "
            "but symbol, data span, and stride match (e.g. dt_v2)"
        ),
    )

    @field_validator("label_cache_suffix")
    @classmethod
    def _label_cache_suffix_safe(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError(
                "label_cache_suffix: use letters, digits, underscore, hyphen only",
            )
        return v

    max_day_trades: int = Field(default=3, ge=1, le=3)
    rolling_business_days: int = Field(default=5, ge=1)

    # Ingest / TwelveData (Iteration 2)
    ingest_chunk_calendar_days: int = Field(
        default=10,
        ge=1,
        le=31,
        description="Split historical requests into windows of this many calendar days",
    )
    twelvedata_outputsize: int = Field(
        default=5000,
        ge=100,
        le=5000,
        description="Max bars per API request (TwelveData cap is 5000)",
    )
    http_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0)
    retry_max_attempts: int = Field(default=6, ge=1, le=20)
    twelvedata_exchange: str | None = Field(
        default=None,
        description="Optional exchange id for TwelveData, e.g. NASDAQ",
    )
    ingest_sleep_seconds_between_chunks: float = Field(
        default=20.0,
        ge=0.0,
        le=300.0,
        description="Pause between chunk requests (free tier ~8 API credits/minute)",
    )
    twelvedata_per_minute_credit_wait_seconds: float = Field(
        default=65.0,
        ge=30.0,
        le=300.0,
        description="Sleep when API reports per-minute credit exhaustion, before retry",
    )

    train_start: date | None = None
    train_end: date | None = None
    val_start: date | None = None
    val_end: date | None = None

    model: ModelConfig = Field(default_factory=lambda: ModelConfig())
    preprocess: PreprocessConfig = Field(default_factory=lambda: PreprocessConfig())
    train: TrainConfig = Field(default_factory=lambda: TrainConfig())
    features: FeatureConfig = Field(default_factory=lambda: FeatureConfig())
    journal: JournalConfig = Field(default_factory=lambda: JournalConfig())
    paths: PathsConfig = Field(default_factory=lambda: PathsConfig())
    context_ingest: ContextIngestConfig = Field(
        default_factory=lambda: ContextIngestConfig(),
        description="Optional SPY/VIX (etc.) cache for market_context features",
    )
    live_ingest: LiveIngestConfig = Field(
        default_factory=lambda: LiveIngestConfig(),
        description="Phase 2 near-live refresh (Plan A); disabled by default",
    )

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
        if self.features.market_context:
            tickers = {s.symbol.upper().lstrip("^") for s in self.context_ingest.symbols}
            if "SPY" not in tickers:
                raise ValueError(
                    "features.market_context requires SPY (1min) in context_ingest",
                )
            vol = [
                s
                for s in self.context_ingest.symbols
                if s.interval == "1day" and s.symbol.upper().lstrip("^") != "SPY"
            ]
            if not vol:
                raise ValueError(
                    "features.market_context requires a 1day volatility proxy in "
                    "context_ingest (TwelveData has no spot VIX/^VIX; use VIXY with "
                    "twelvedata_exchange: CBOE)",
                )
        return self
