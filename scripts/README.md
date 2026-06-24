# Scripts

Utilities that are **not** part of the installed `sparkles` package. Run from the **repository root**:

```bash
python scripts/quick_try_vol.py -c configs/experiments/rklb_baseline.yaml
python scripts/run_trials.py --dry-run
python scripts/run_trials.py
python scripts/run_grid_search.py --dry-run --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
python scripts/run_grid_search.py --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
```

For the main pipeline, use the **`sparkles`** CLI (see **README.md** and **METHODOLOGY.md**).

**`run_trials.py`** — batch hyperparameter trials: merges **`configs/experiments/presets/*.yaml`** onto a base experiment, optional **`--dry-run`**, then trains and exports **`artifacts/training_log.csv`**. See **`configs/experiments/presets/README.md`**.

**`run_grid_search.py`** — cartesian grid over dotted YAML paths (model, features, train knobs). Each invocation writes a timestamped folder under **`artifacts/grid_search/`** (`dry_run_log.txt`, `dry_run_summary.csv`, or `results.csv` + `train_log.txt`). Progress prints every **`--progress-every N`** combos (default 100).
