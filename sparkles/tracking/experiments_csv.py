"""Flatten ``experiments.jsonl`` rows to a wide CSV for spreadsheets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sparkles.config.schema import ExperimentConfig


def flatten_log_row(record: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Turn a nested JSON object into a single-level dict with dotted keys."""

    def walk(obj: Any, prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                out.update(walk(v, key))
        elif isinstance(obj, (list, tuple)):
            out[prefix] = json.dumps(obj, default=str)
        elif obj is None:
            out[prefix] = ""
        elif isinstance(obj, bool):
            out[prefix] = obj
        else:
            out[prefix] = obj
        return out

    return walk(record, "")


def export_experiments_to_csv(
    log_path: Path,
    output_path: Path,
    *,
    symbol_filter: str | None = None,
) -> int:
    """Read JSONL, optionally filter by ``symbol`` (uppercase), write CSV.

    Returns number of rows written.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Experiment log not found: {log_path}")

    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if symbol_filter is not None:
            if str(rec.get("symbol", "")).upper() != symbol_filter.upper():
                continue
        rows.append(flatten_log_row(rec))

    if not rows:
        # Still write a header-only file from an empty frame for predictable tooling
        pd.DataFrame().to_csv(output_path, index=False)
        return 0

    df = pd.DataFrame(rows)
    priority = ["logged_at_utc", "run_id", "symbol", "val_accuracy"]
    first = [c for c in priority if c in df.columns]
    rest = sorted(c for c in df.columns if c not in first)
    df = df[first + rest]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)


def experiments_log_path(cfg: ExperimentConfig, base_dir: Path | None = None) -> Path:
    root = Path.cwd() if base_dir is None else base_dir
    return root / cfg.paths.artifacts_dir / "experiments.jsonl"
