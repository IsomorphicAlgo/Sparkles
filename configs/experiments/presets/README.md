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
| **`rklb_daytrade_g1_v1.yaml`** | G1 on **`rklb_daytrade_v1.yaml`** (day-trade labels; label first) |

**Day-trade experiment (new labels):**

```bash
# v2 — intraday barriers (recommended after v1 vertical-only collapse)
sparkles label -c configs/experiments/rklb_daytrade_v2.yaml
sparkles train -c configs/experiments/rklb_daytrade_v2.yaml --dry-run
python scripts/run_trials.py --base configs/experiments/rklb_daytrade_v2.yaml --preset configs/experiments/presets/rklb_daytrade_v2_g1.yaml
```

v1 (15%/10% barriers) → `…_s15.parquet`; v2 → `…_s15_dt_v2.parquet` (does not overwrite v1).

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
