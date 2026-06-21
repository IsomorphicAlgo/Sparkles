"""Context symbol helpers (Phase G3); ingest logic lives in ``ingest.py``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sparkles.data.ingest import run_symbol_ingest, symbol_parquet_path

if TYPE_CHECKING:
    from sparkles.config.schema import ContextSymbolConfig, ExperimentConfig


def context_parquet_path(
    cfg: ExperimentConfig,
    spec: ContextSymbolConfig,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Cache path for a context symbol over the experiment data span."""
    return symbol_parquet_path(cfg, spec.symbol, spec.interval, base_dir=base_dir)


def run_context_symbol_ingest(
    cfg: ExperimentConfig,
    spec: ContextSymbolConfig,
    *,
    api_key: str | None = None,
    force_refresh: bool = False,
    base_dir: Path | None = None,
) -> Path:
    """Download one context symbol (same date span as main symbol)."""
    return run_symbol_ingest(
        cfg,
        symbol=spec.symbol,
        interval=spec.interval,
        exchange=spec.twelvedata_exchange,
        api_key=api_key,
        force_refresh=force_refresh,
        base_dir=base_dir,
    )


def run_context_ingest(
    cfg: ExperimentConfig,
    *,
    api_key: str | None = None,
    force_refresh: bool = False,
    base_dir: Path | None = None,
) -> list[Path]:
    """Download every ``context_ingest.symbols`` entry (explicit batch helper)."""
    return [
        run_context_symbol_ingest(
            cfg,
            spec,
            api_key=api_key,
            force_refresh=force_refresh,
            base_dir=base_dir,
        )
        for spec in cfg.context_ingest.symbols
    ]
