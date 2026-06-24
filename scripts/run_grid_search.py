#!/usr/bin/env python3
"""Cartesian grid search over experiment YAML knobs (ML expansion Phase E).

Loads a grid spec YAML, merges base (+ optional preset), trains every combination,
and writes artifacts under ``artifacts/grid_search/{run_id}_{prefix}/``.

Usage (from repository root):

    python scripts/run_grid_search.py --dry-run \\
      --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml

    python scripts/run_grid_search.py \\
      --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparkles.config.grid import build_grid_configs, load_grid_spec
from sparkles.config.grid_runner import (
    default_progress,
    new_grid_run_dir,
    run_grid_dry_run,
    run_grid_train,
    write_grid_meta,
)
from sparkles.config.schema import ExperimentConfig
from sparkles.tracking.experiments_csv import (
    export_experiments_to_csv,
    experiments_log_path,
)

DEFAULT_BASE = REPO_ROOT / "configs" / "experiments" / "rklb_daytrade_v2.yaml"
DEFAULT_GRID_ROOT = REPO_ROOT / "artifacts" / "grid_search"


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
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N combinations (default 100)",
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
    prefix = str(spec.get("experiment_name_prefix") or spec.get("grid_name") or "grid")

    pairs = build_grid_configs(
        spec,
        base_path=base_path.resolve(),
        preset_path=preset.resolve() if preset else None,
    )
    if not pairs:
        print("Grid spec produced zero combinations.", file=sys.stderr)
        return 1

    run_dir = new_grid_run_dir(DEFAULT_GRID_ROOT, prefix=prefix)
    write_grid_meta(
        run_dir,
        {
            "grid_spec": str(args.grid.resolve()),
            "base": str(base_path.resolve()),
            "preset": str(preset.resolve()) if preset else None,
            "n_combinations": len(pairs),
            "dry_run": args.dry_run,
        },
    )

    print(f"Grid: {args.grid.name} — {len(pairs)} combination(s)")
    print(f"Base: {base_path.resolve()}")
    if preset:
        print(f"Preset: {preset.resolve()}")
    print(f"Output: {run_dir.resolve()}")

    last_cfg: ExperimentConfig | None = None
    ok_all = True

    if args.dry_run:
        df, ready_n = run_grid_dry_run(
            pairs,
            run_dir,
            base_dir=REPO_ROOT,
            progress_every=args.progress_every,
            progress=default_progress,
        )
        ok_all = ready_n == len(pairs)
        print(f"\nDry-run ready: {ready_n}/{len(pairs)}")
        print(f"Log:  {run_dir / 'dry_run_log.txt'}")
        print(f"CSV:  {run_dir / 'dry_run_summary.csv'}")
        last_cfg = pairs[-1][1] if pairs else None
    else:
        df, best = run_grid_train(
            pairs,
            run_dir,
            base_dir=REPO_ROOT,
            progress_every=args.progress_every,
            progress=default_progress,
        )
        ok_all = bool(df["ok"].all()) if "ok" in df.columns and len(df) else True
        print(f"\nWrote {len(df)} row(s) to {run_dir / 'results.csv'}")
        print(f"Log: {run_dir / 'train_log.txt'}")
        if best:
            print(
                f"Best val_f1_macro={best.get('val_f1_macro')}  "
                f"run_id={best.get('run_id')}",
            )
        last_cfg = pairs[-1][1] if pairs else None

    if (
        not args.dry_run
        and not args.no_export_log
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
