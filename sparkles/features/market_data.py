"""Load context-symbol Parquet for market_context features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sparkles.config.schema import ContextSymbolConfig, ExperimentConfig
from sparkles.data.ingest import load_symbol_ohlcv
from sparkles.data.symbol_hints import VOLATILITY_PROXY_HINT, validate_volatility_proxy_symbol

# TwelveData does not expose CBOE spot VIX (^VIX) on most plans; use a listed proxy.


def _spec_for_ticker(cfg: ExperimentConfig, ticker: str) -> ContextSymbolConfig | None:
    want = ticker.upper().lstrip("^")
    for spec in cfg.context_ingest.symbols:
        if spec.symbol.upper().lstrip("^") == want:
            return spec
    return None


def volatility_proxy_spec(cfg: ExperimentConfig) -> ContextSymbolConfig | None:
    """First non-SPY 1day symbol in context_ingest (VIXY, VXX, etc.)."""
    for spec in cfg.context_ingest.symbols:
        if spec.interval == "1day" and spec.symbol.upper().lstrip("^") != "SPY":
            return spec
    return None


def load_context_frame(
    cfg: ExperimentConfig,
    ticker: str,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """Read cached context Parquet for ``ticker`` (raises if missing)."""
    validate_volatility_proxy_symbol(ticker)
    spec = _spec_for_ticker(cfg, ticker)
    if spec is None:
        raise ValueError(
            f"context_ingest has no entry for {ticker.upper()!r}; "
            "add it to the experiment YAML and run sparkles ingest",
        )
    return load_symbol_ohlcv(cfg, spec.symbol, spec.interval, base_dir=base_dir)


def load_market_context_frames(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (SPY 1m OHLCV, daily volatility-proxy OHLCV) for market_context."""
    vol_spec = volatility_proxy_spec(cfg)
    if vol_spec is None:
        raise ValueError(
            "features.market_context requires a 1day volatility proxy in "
            f"context_ingest (not SPY). {VOLATILITY_PROXY_HINT}",
        )
    spy = load_context_frame(cfg, "SPY", base_dir=base_dir)
    vol = load_context_frame(cfg, vol_spec.symbol, base_dir=base_dir)
    return spy, vol
