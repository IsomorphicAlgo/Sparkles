"""Append-only JSONL run log under the artifacts root (Iteration 6)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_experiment_record(
    artifacts_root: Path,
    record: dict[str, Any],
) -> Path:
    """Append one JSON object per line to ``artifacts_root/experiments.jsonl``."""
    artifacts_root.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_root / "experiments.jsonl"
    line = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return log_path
