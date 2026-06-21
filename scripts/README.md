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

**`run_grid_search.py`** — cartesian grid over dotted YAML paths (model, features, train knobs). Spec YAML under **`configs/experiments/grids/`**; writes **`artifacts/grid_search/*.csv`** and appends each run to **`experiments.jsonl`**. Use **`fixed.train.export_predictions: none`** in the spec to speed up large grids.
