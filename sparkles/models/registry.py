"""Versioned artifact paths: ``artifacts/{SYMBOL}/{run_id}/`` (Iteration 6)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from sparkles.config.schema import ExperimentConfig


def new_run_id() -> str:
    """UTC timestamp suitable for directory names (microsecond resolution)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def run_artifact_dir(
    cfg: ExperimentConfig,
    run_id: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Create and return ``{artifacts_dir}/{SYMBOL}/{run_id}/``."""
    root = Path.cwd() if base_dir is None else base_dir
    p = root / cfg.paths.artifacts_dir / cfg.symbol.upper() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    """Persist sklearn objects + metadata via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def save_json(path: Path, obj: Any) -> None:
    """Write JSON (metrics, config snapshot)."""
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
