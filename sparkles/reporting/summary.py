"""Phase 1 artifact / cache summary for ``sparkles report`` (Iteration 7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.labels.triple_barrier import labeled_parquet_path


def _runs_with_metrics(symbol_dir: Path) -> list[Path]:
    if not symbol_dir.is_dir():
        return []
    out: list[Path] = []
    for p in symbol_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").is_file():
            out.append(p)
    return sorted(out, key=lambda x: x.name, reverse=True)


def _tail_experiments_for_symbol(
    log_path: Path,
    symbol: str,
    *,
    max_entries: int = 5,
) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    symu = symbol.upper()
    raw = log_path.read_text(encoding="utf-8").splitlines()
    picked: list[dict[str, Any]] = []
    for line in reversed(raw[-500:]):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(obj.get("symbol", "")).upper() == symu:
            picked.append(obj)
            if len(picked) >= max_entries:
                break
    return list(reversed(picked))


def run_phase1_report(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    run_id: str | None = None,
) -> str:
    """Build a multi-line human-readable summary (no trailing newline required)."""
    root = Path.cwd() if base_dir is None else base_dir
    lines: list[str] = []

    lines.append(
        f"experiment: symbol={cfg.symbol.upper()}  "
        f"data_range={cfg.data_start}..{cfg.data_end}",
    )

    ingest_p = parquet_cache_path(cfg, base_dir=base_dir)
    label_p = labeled_parquet_path(cfg, base_dir=base_dir)
    ir = ingest_p.resolve()
    lr = label_p.resolve()
    lines.append(f"ingest_cache: exists={ingest_p.is_file()}  path={ir}")
    lines.append(f"labeled:     exists={label_p.is_file()}  path={lr}")

    art_sym = root / cfg.paths.artifacts_dir / cfg.symbol.upper()
    metrics_path: Path | None = None
    if run_id:
        cand = art_sym / run_id
        if (cand / "metrics.json").is_file():
            metrics_path = cand / "metrics.json"
            lines.append(f"train_run:   (explicit) {cand.resolve()}")
        else:
            lines.append(
                f"train_run:   --run {run_id!r} not found or missing metrics.json",
            )
    else:
        runs = _runs_with_metrics(art_sym)
        if runs:
            metrics_path = runs[0] / "metrics.json"
            lines.append(f"train_run:   latest {runs[0].resolve()}")
        else:
            lines.append(
                "train_run:   (none — run sparkles train after ingest and label)",
            )

    if metrics_path is not None:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.append(
            f"  metrics: train_acc={m.get('train_accuracy')}  "
            f"val_acc={m.get('val_accuracy')}  "
            f"train_n={m.get('train_n')}  val_n={m.get('val_n')}",
        )

    exp_log = root / cfg.paths.artifacts_dir / "experiments.jsonl"
    tail = _tail_experiments_for_symbol(exp_log, cfg.symbol)
    if tail:
        er = exp_log.resolve()
        lines.append(f"experiments_log: {er} (last {len(tail)} for symbol)")
        for row in tail:
            rid = row.get("run_id", "?")
            va = row.get("val_accuracy", "?")
            lines.append(f"  {rid}  val_accuracy={va}")
    elif exp_log.is_file():
        er = exp_log.resolve()
        lines.append(f"experiments_log: {er} (no rows for this symbol yet)")
    else:
        lines.append("experiments_log: (file not created yet)")

    return "\n".join(lines)
