# Scripts

Utilities that are **not** part of the installed `sparkles` package. Run from the **repository root**:

```bash
python scripts/quick_try_vol.py -c configs/experiments/rklb_baseline.yaml
python scripts/run_trials.py --dry-run
python scripts/run_trials.py
```

For the main pipeline, use the **`sparkles`** CLI (see **README.md** and **METHODOLOGY.md**).

**`run_trials.py`** — batch hyperparameter trials: merges **`configs/experiments/presets/*.yaml`** onto a base experiment, optional **`--dry-run`**, then trains and exports **`artifacts/training_log.csv`**. See **`configs/experiments/presets/README.md`**.
