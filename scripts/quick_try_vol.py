"""Load cached 1m Parquet, attach volatility columns, print a short summary.

Run from repo root:
    python scripts/quick_try_vol.py
    python scripts/quick_try_vol.py --config configs/experiments/rklb_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparkles.config import load_experiment_config
from sparkles.data.ingest import parquet_cache_path
from sparkles.features import add_volatility_from_config


def main() -> None:
    desc = "Attach vol columns to cached 1m Parquet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/experiments/rklb_baseline.yaml"),
        help="Experiment YAML",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    pq = parquet_cache_path(cfg)
    if not pq.is_file():
        print(f"No Parquet at {pq.resolve()}")
        print(f"Run first: sparkles ingest -c {args.config}")
        raise SystemExit(1)

    df = pd.read_parquet(pq)
    out = add_volatility_from_config(df, cfg)
    ann = f"vol_{cfg.vol_lookback_trading_days}d_ann"
    daily = f"sigma_daily_{cfg.vol_lookback_trading_days}d"
    print(f"Rows: {len(out)}  Parquet: {pq.resolve()}")
    print(out[[c for c in ("close", daily, ann) if c in out.columns]].tail(8))
    print()
    print(out[ann].describe())


if __name__ == "__main__":
    main()
