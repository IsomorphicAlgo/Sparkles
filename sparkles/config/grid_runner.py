"""Run grid dry-run / train loops with file logs and quiet progress."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sparkles.config.schema import ExperimentConfig
from sparkles.models.registry import new_run_id
from sparkles.models.train import (
    TrainDryRunReport,
    dry_run_train,
    format_dry_run_report,
    run_train,
)

ProgressCallback = Callable[[int, int, str], None]

DEFAULT_PROGRESS_EVERY = 100


def new_grid_run_dir(base_dir: Path, *, prefix: str = "grid") -> Path:
    """Create ``artifacts/grid_search/{run_id}_{prefix}/`` for one grid session."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)
    run_dir = base_dir / f"{new_run_id()}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_grid_meta(run_dir: Path, meta: dict[str, Any]) -> Path:
    """Write ``meta.json`` describing the grid session."""
    path = run_dir / "meta.json"
    payload = {"written_at_utc": datetime.now(UTC).isoformat(), **meta}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def default_progress(i: int, n: int, label: str = "") -> None:
    """Print a single progress line (safe for large grids)."""
    pct = 100.0 * i / n if n else 100.0
    suffix = f"  {label}" if label else ""
    print(f"[{i}/{n}] ({pct:.1f}%){suffix}", flush=True)


def maybe_progress(
    i: int,
    n: int,
    *,
    every: int,
    callback: ProgressCallback | None,
    label: str = "",
) -> None:
    if callback is None:
        return
    if i == 1 or i == n or (every > 0 and i % every == 0):
        callback(i, n, label)


@contextmanager
def _quiet_sparkles_logs():
    targets = [
        logging.getLogger("sparkles"),
        logging.getLogger("sparkles.models"),
        logging.getLogger("sparkles.models.train"),
        logging.getLogger("sparkles.features"),
    ]
    saved = [(lg, lg.level, lg.propagate) for lg in targets]
    for lg in targets:
        lg.setLevel(logging.ERROR)
        lg.propagate = False
    try:
        yield
    finally:
        for lg, level, propagate in saved:
            lg.setLevel(level)
            lg.propagate = propagate


def _combo_header(i: int, n: int, cfg: ExperimentConfig, combo: dict[str, Any]) -> str:
    name = cfg.train.experiment_name or f"grid_{i}"
    lines = [f"--- [{i}/{n}] {name} ---"]
    for path, value in sorted(combo.items()):
        lines.append(f"  {path}: {value}")
    return "\n".join(lines)


def _dry_run_row(
    combo: dict[str, Any],
    cfg: ExperimentConfig,
    report: TrainDryRunReport,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ready": report.ready,
        "experiment_name": cfg.train.experiment_name,
        "train_n": report.train_n,
        "val_n": report.val_n,
        "val_rows_dropped_unseen": report.val_rows_dropped_unseen,
        "sample_weight_method": report.sample_weight_method,
        "n_feature_columns": len(report.feature_columns),
        "error": "; ".join(report.issues) if report.issues else "",
    }
    for path, value in combo.items():
        row[f"grid.{path}"] = value
    return row


def _train_row(
    combo: dict[str, Any],
    cfg: ExperimentConfig,
    *,
    ok: bool,
    run_dir: Path | None,
    error: str | None,
    elapsed_s: float,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ok": ok,
        "elapsed_s": round(elapsed_s, 3),
        "experiment_name": cfg.train.experiment_name,
        "run_dir": str(run_dir.resolve()) if run_dir else "",
        "error": error or "",
    }
    for path, value in combo.items():
        row[f"grid.{path}"] = value
    if metrics:
        row["run_id"] = run_dir.name if run_dir else ""
        for key in (
            "val_f1_macro",
            "val_f1_weighted",
            "val_accuracy",
            "train_f1_macro",
            "train_accuracy",
            "train_n",
            "val_n",
            "sample_weight_method",
            "sample_weight_mean",
        ):
            row[key] = metrics.get(key)
    return row


def _order_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    grid_cols = sorted(c for c in df.columns if c.startswith("grid."))
    priority = [
        "ready",
        "ok",
        "val_f1_macro",
        "val_f1_weighted",
        "val_accuracy",
        "train_f1_macro",
        "elapsed_s",
        "experiment_name",
        "run_id",
        "train_n",
        "val_n",
        "error",
        "run_dir",
    ]
    metric_cols = [c for c in priority if c in df.columns]
    other = [c for c in df.columns if c not in grid_cols + metric_cols]
    return df[grid_cols + metric_cols + other]


def run_grid_dry_run(
    pairs: Sequence[tuple[dict[str, Any], ExperimentConfig]],
    run_dir: Path,
    *,
    base_dir: Path | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    progress: ProgressCallback | None = default_progress,
) -> tuple[pd.DataFrame, int]:
    """Dry-run every combo; verbose text → ``dry_run_log.txt``, summary → ``dry_run_summary.csv``."""
    n = len(pairs)
    log_path = run_dir / "dry_run_log.txt"
    csv_path = run_dir / "dry_run_summary.csv"
    rows: list[dict[str, Any]] = []
    ready_n = 0

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"Grid dry-run — {n} combination(s)\n")
        log_f.write(f"Output dir: {run_dir.resolve()}\n\n")

        with _quiet_sparkles_logs():
            for i, (combo, cfg) in enumerate(pairs, start=1):
                maybe_progress(
                    i,
                    n,
                    every=progress_every,
                    callback=progress,
                    label="dry-run",
                )
                log_f.write(_combo_header(i, n, cfg, combo))
                log_f.write("\n")
                report = dry_run_train(cfg, base_dir=base_dir)
                log_f.write(format_dry_run_report(report))
                log_f.write("\n\n")
                rows.append(_dry_run_row(combo, cfg, report))
                if report.ready:
                    ready_n += 1

    df = _order_grid_columns(pd.DataFrame(rows))
    df.to_csv(csv_path, index=False)
    return df, ready_n


def run_grid_train(
    pairs: Sequence[tuple[dict[str, Any], ExperimentConfig]],
    run_dir: Path,
    *,
    base_dir: Path | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    progress: ProgressCallback | None = default_progress,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Train every combo; one-line status → ``train_log.txt``, metrics → ``results.csv``."""
    n = len(pairs)
    log_path = run_dir / "train_log.txt"
    csv_path = run_dir / "results.csv"
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"Grid train — {n} combination(s)\n")
        log_f.write(f"Output dir: {run_dir.resolve()}\n\n")
        t_all = time.perf_counter()

        with _quiet_sparkles_logs():
            for i, (combo, cfg) in enumerate(pairs, start=1):
                name = cfg.train.experiment_name or f"grid_{i}"
                maybe_progress(
                    i,
                    n,
                    every=progress_every,
                    callback=progress,
                    label=name[:60],
                )
                t0 = time.perf_counter()
                try:
                    out_dir = run_train(cfg, base_dir=base_dir)
                    metrics = json.loads(
                        (out_dir / "metrics.json").read_text(encoding="utf-8"),
                    )
                    elapsed = time.perf_counter() - t0
                    row = _train_row(
                        combo,
                        cfg,
                        ok=True,
                        run_dir=out_dir,
                        error=None,
                        elapsed_s=elapsed,
                        metrics=metrics,
                    )
                    rows.append(row)
                    log_f.write(
                        f"[{i}/{n}] ok  val_f1_macro={row.get('val_f1_macro')}  "
                        f"elapsed={elapsed:.2f}s  {name}\n",
                    )
                    vf1 = row.get("val_f1_macro")
                    if vf1 is not None and (
                        best is None or vf1 > best.get("val_f1_macro", -1)
                    ):
                        best = row
                except (FileNotFoundError, ValueError, KeyError) as e:
                    elapsed = time.perf_counter() - t0
                    row = _train_row(
                        combo,
                        cfg,
                        ok=False,
                        run_dir=None,
                        error=str(e),
                        elapsed_s=elapsed,
                    )
                    rows.append(row)
                    log_f.write(f"[{i}/{n}] FAIL  {e}  {name}\n")

        log_f.write(f"\nTotal elapsed: {time.perf_counter() - t_all:.1f}s\n")
        if best:
            log_f.write(
                f"Best val_f1_macro={best.get('val_f1_macro')}  "
                f"run_id={best.get('run_id')}\n",
            )

    df = _order_grid_columns(pd.DataFrame(rows))
    if "val_f1_macro" in df.columns:
        df = df.sort_values("val_f1_macro", ascending=False, na_position="last")
    df.to_csv(csv_path, index=False)
    return df, best
