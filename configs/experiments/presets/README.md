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
