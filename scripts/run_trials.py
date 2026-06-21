#!/usr/bin/env python3
"""Run a batch of experiment preset overlays (ML expansion Phase E).

Merge each preset under ``configs/experiments/presets/`` onto a base YAML,
optionally dry-run, then train. Writes ``experiments.jsonl`` like normal
``sparkles train`` and optionally exports a wide CSV at the end.

Usage (from repository root):

    python scripts/run_trials.py --dry-run
    python scripts/run_trials.py
    python scripts/run_trials.py --preset configs/experiments/presets/xgb_shallow.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparkles.config.load import load_experiment_config_merged
from sparkles.config.schema import ExperimentConfig
from sparkles.models.train import dry_run_train, format_dry_run_report, run_train
from sparkles.tracking.experiments_csv import (
    export_experiments_to_csv,
    experiments_log_path,
)

DEFAULT_BASE = REPO_ROOT / "configs" / "experiments" / "rklb_baseline.yaml"
DEFAULT_PRESETS_DIR = REPO_ROOT / "configs" / "experiments" / "presets"
DEFAULT_CSV = REPO_ROOT / "artifacts" / "training_log.csv"


def discover_presets(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.glob("*.yaml")
        if p.is_file()
    )


def run_one(
    cfg: ExperimentConfig,
    *,
    dry_run: bool,
) -> tuple[bool, str]:
    if dry_run:
        report = dry_run_train(cfg)
        text = format_dry_run_report(report)
        return report.ready, text
    try:
        out = run_train(cfg)
    except (FileNotFoundError, ValueError, KeyError) as e:
        return False, str(e)
    return True, str(out.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch train experiment presets.")
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base experiment YAML (default: {DEFAULT_BASE.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--presets-dir",
        type=Path,
        default=DEFAULT_PRESETS_DIR,
        help="Directory of overlay YAML files",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        action="append",
        dest="presets",
        help="Single preset overlay (repeatable); default: all *.yaml in presets-dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pre-flight only; do not fit models",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip CSV export after successful trains",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV path when exporting (default: {DEFAULT_CSV.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    preset_paths = args.presets or discover_presets(args.presets_dir)
    if not preset_paths:
        print(f"No preset YAML files found in {args.presets_dir}", file=sys.stderr)
        return 1

    ok_all = True
    last_cfg: ExperimentConfig | None = None
    for preset in preset_paths:
        print(f"\n=== {preset.name} ===")
        cfg = load_experiment_config_merged(args.base, preset)
        last_cfg = cfg
        ok, msg = run_one(cfg, dry_run=args.dry_run)
        print(msg)
        if not ok:
            ok_all = False

    if args.dry_run or args.no_export or not ok_all or last_cfg is None:
        return 0 if ok_all else 1

    log_path = experiments_log_path(last_cfg, base_dir=REPO_ROOT)
    n = export_experiments_to_csv(
        log_path,
        args.output.resolve(),
        symbol_filter=last_cfg.symbol.upper(),
    )
    print(f"\nExported {n} row(s) to {args.output.resolve()}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
