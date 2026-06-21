#!/usr/bin/env python3
"""Cartesian grid search over experiment YAML knobs (ML expansion Phase E).

Loads a grid spec YAML, merges base (+ optional preset), trains every combination,
and writes a wide results CSV. Each run also appends to ``artifacts/experiments.jsonl``.

Usage (from repository root):

    python scripts/run_grid_search.py --dry-run \\
      --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml

    python scripts/run_grid_search.py \\
      --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparkles.config.grid import build_grid_configs, load_grid_spec
from sparkles.config.schema import ExperimentConfig
from sparkles.models.train import dry_run_train, format_dry_run_report, run_train
from sparkles.tracking.experiments_csv import (
    export_experiments_to_csv,
    experiments_log_path,
)

DEFAULT_BASE = REPO_ROOT / "configs" / "experiments" / "rklb_daytrade_v2.yaml"
DEFAULT_CSV = REPO_ROOT / "artifacts" / "grid_search_results.csv"


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _result_row(
    combo: dict[str, Any],
    cfg: ExperimentConfig,
    *,
    ok: bool,
    run_dir: Path | None,
    error: str | None,
    elapsed_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ok": ok,
        "elapsed_s": round(elapsed_s, 3),
        "experiment_name": cfg.train.experiment_name,
        "run_dir": str(run_dir) if run_dir else "",
        "error": error or "",
    }
    for path, value in combo.items():
        row[f"grid.{path}"] = value
    if run_dir is not None:
        row["run_id"] = run_dir.name
        metrics = _read_metrics(run_dir)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grid search over experiment YAML knobs.")
    parser.add_argument(
        "--grid",
        type=Path,
        required=True,
        help="Grid spec YAML (see configs/experiments/grids/)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base experiment YAML (overrides spec.base when set)",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Preset overlay YAML (overrides spec.preset when set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pre-flight each combo; do not fit models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Results CSV path (default: spec.output or artifacts/grid_search_results.csv)",
    )
    parser.add_argument(
        "--no-export-log",
        action="store_true",
        help="Skip refreshing artifacts/training_log.csv after a real train grid",
    )
    args = parser.parse_args(argv)

    spec = load_grid_spec(args.grid.resolve())
    base_path = args.base or Path(spec.get("base") or DEFAULT_BASE)
    preset_path = args.preset or spec.get("preset")
    preset = Path(preset_path) if preset_path else None

    pairs = build_grid_configs(
        spec,
        base_path=base_path.resolve(),
        preset_path=preset.resolve() if preset else None,
    )
    if not pairs:
        print("Grid spec produced zero combinations.", file=sys.stderr)
        return 1

    print(f"Grid: {args.grid.name} — {len(pairs)} combination(s)")
    print(f"Base: {base_path.resolve()}")
    if preset:
        print(f"Preset: {preset.resolve()}")

    rows: list[dict[str, Any]] = []
    ok_all = True
    last_cfg: ExperimentConfig | None = None

    for i, (combo, cfg) in enumerate(pairs, start=1):
        name = cfg.train.experiment_name or f"grid_{i}"
        print(f"\n=== [{i}/{len(pairs)}] {name} ===")
        for path, value in sorted(combo.items()):
            print(f"  {path}: {value}")

        t0 = time.perf_counter()
        if args.dry_run:
            report = dry_run_train(cfg)
            text = format_dry_run_report(report)
            print(text)
            ok = report.ready
            err = None if ok else "dry-run not ready"
            run_dir = None
        else:
            try:
                run_dir = run_train(cfg)
                ok = True
                err = None
            except (FileNotFoundError, ValueError, KeyError) as e:
                ok = False
                err = str(e)
                run_dir = None
                print(err, file=sys.stderr)

        elapsed = time.perf_counter() - t0
        rows.append(
            _result_row(combo, cfg, ok=ok, run_dir=run_dir, error=err, elapsed_s=elapsed),
        )
        last_cfg = cfg
        if not ok:
            ok_all = False

    out_path = args.output or spec.get("output")
    csv_path = Path(out_path).resolve() if out_path else DEFAULT_CSV.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    grid_cols = sorted(c for c in df.columns if c.startswith("grid."))
    metric_cols = [
        c
        for c in (
            "ok",
            "elapsed_s",
            "experiment_name",
            "run_id",
            "val_f1_macro",
            "val_f1_weighted",
            "val_accuracy",
            "train_f1_macro",
            "sample_weight_method",
            "run_dir",
            "error",
        )
        if c in df.columns
    ]
    other = [c for c in df.columns if c not in grid_cols + metric_cols]
    df = df[grid_cols + metric_cols + other]
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} row(s) to {csv_path}")

    if (
        not args.dry_run
        and not args.no_export_log
        and ok_all
        and last_cfg is not None
    ):
        log_path = experiments_log_path(last_cfg, base_dir=REPO_ROOT)
        n = export_experiments_to_csv(
            log_path,
            (REPO_ROOT / "artifacts" / "training_log.csv").resolve(),
            symbol_filter=last_cfg.symbol.upper(),
        )
        print(f"Refreshed artifacts/training_log.csv ({n} row(s))")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
