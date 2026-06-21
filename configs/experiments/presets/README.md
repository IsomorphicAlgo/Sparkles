# Experiment presets (ML expansion Phase E)

These files are **overlays** merged on top of a base experiment YAML (usually
`configs/experiments/rklb_baseline.yaml`). They change **`model:`** and **`train:`**
only so you can run hyperparameter trials without duplicating ingest/label settings.

**Dry-run one preset:**

```bash
python scripts/run_trials.py --base configs/experiments/rklb_baseline.yaml --dry-run
```

**Train all presets and refresh the comparison CSV:**

```bash
python scripts/run_trials.py --base configs/experiments/rklb_baseline.yaml
```

**Single preset via CLI** (merge in Python, then train):

```bash
sparkles train -c configs/experiments/rklb_baseline.yaml --dry-run
# Edit rklb_baseline.yaml, or use run_trials.py for preset overlays.
```

| Preset | Intent |
|--------|--------|
| `logistic_balanced.yaml` | Logistic regression + `class_weight: balanced` |
| `xgb_shallow.yaml` | XGBoost, shallow trees (needs `pip install -e ".[ml]"`) |
| `xgb_deep.yaml` | XGBoost, deeper + slower learning rate |
| **`xgb_d3_reg_v1.yaml`** | **Champion** (2026-06-20): depth 3, lr 0.08, n_estimators 60, subsample/colsample 0.8 — best val F1 so far |
| **`g1_features_v1.yaml`** | **Phase G1** features + champion XGB hyperparams — A/B vs baseline |
| **`rklb_daytrade_g1_v1.yaml`** | G1 on **`rklb_daytrade_v1.yaml`** (day-trade v1 labels; label first) |
| **`rklb_daytrade_g1_g2_g3_v1.yaml`** | G1 + G2 + G3 on day-trade v2 (feature flags only; base model hparams) |
| **`rklb_daytrade_champion_v1.yaml`** | **Day-trade champion** (2026-06-21): G1+G2+G3 + tuned XGB (depth 3, lr 0.08, n_estimators 127) — reproduces **`Trial_RB_G1_G2_G3_v1`** |
| **`rklb_daytrade_champion_uniqueness_v1.yaml`** | Champion v1 + **`train.sample_weight_method: uniqueness`** (I4 AFML weights) |

**Day-trade experiment (new labels):**

```bash
# v2 — intraday barriers (recommended after v1 vertical-only collapse)
sparkles label -c configs/experiments/rklb_daytrade_v2.yaml
sparkles train -c configs/experiments/rklb_daytrade_v2.yaml --dry-run
python scripts/run_trials.py --base configs/experiments/rklb_daytrade_v2.yaml --preset configs/experiments/presets/rklb_daytrade_v2_g1.yaml
```

v1 (15%/10% barriers) → `…_s15.parquet`; v2 → `…_s10_dt_v2.parquet` (stride 10 in current v2 YAML).

**Champion day-trade preset:**

```bash
sparkles label -c configs/experiments/rklb_daytrade_v2.yaml
sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s SPY -i 1min
sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s VIXY -i 1day
python scripts/run_trials.py --base configs/experiments/rklb_daytrade_v2.yaml \
  --preset configs/experiments/presets/rklb_daytrade_champion_v1.yaml
```

**Champion + I4 uniqueness preset:**

```bash
python scripts/run_trials.py --base configs/experiments/rklb_daytrade_v2.yaml \
  --preset configs/experiments/presets/rklb_daytrade_champion_uniqueness_v1.yaml
```

**Grid search** (cartesian sweep over model/feature/train knobs — see `configs/experiments/grids/`):

```bash
python scripts/run_grid_search.py --dry-run --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
python scripts/run_grid_search.py --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml
```

**Train G1 preset:**

```bash
python scripts/run_trials.py --preset configs/experiments/presets/g1_features_v1.yaml
```

**Train champion preset only:**

```bash
python scripts/run_trials.py --preset configs/experiments/presets/xgb_d3_reg_v1.yaml
sparkles experiments export -c configs/experiments/rklb_baseline.yaml
```

```bash
sparkles experiments export -c configs/experiments/rklb_baseline.yaml
```
