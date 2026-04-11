---
name: ML experimentation expansion
overview: "Living plan to grow YAML-driven experiments: more model types, hyperparameters, feature toggles, and preprocessing—without bloating train.py. Complements Phase 1 (plan.md); work here is incremental and owner-approved when treated as a mini-roadmap."
todos:
  - id: config-surface
    content: "Phase A: TrainConfig + extended ModelConfig (class_weight, solver, tol); richer experiments.jsonl"
    status: not started
  - id: feature-registry
    content: "Phase B: FeatureConfig YAML + named builders in features/; refactor dataset join"
    status: not started
  - id: model-registry
    content: "Phase C: Estimator factory + second model (e.g. xgboost under [ml] extra)"
    status: not started
  - id: preprocessing-pipeline
    content: "Phase D: sklearn Pipeline scaler fit-on-train-only; save in bundle"
    status: not started
  - id: experiment-workflow
    content: "Phase E: train --dry-run, optional sweep script, config presets folder"
    status: not started
  - id: evaluation-metrics
    content: "Phase F: macro/weighted F1, imbalance-aware reporting in CLI/metrics.json"
    status: not started
isProject: false
---

# ML experimentation expansion

This file is the **living roadmap** for improving **models**, **hyperparameters**, and **features** beyond the Phase 1 baseline (`logistic_regression` + fixed entry-only columns). **`plan.md`** remains the authority for **Phase 1** iteration history; this doc tracks **ML engineering expansion** work.

---

## Mandatory habits (agents & humans)

1. **No leakage:** Features must use only information **knowable at `entry_time`** unless a design explicitly documents a different causal window and is reviewed.
2. **YAML first:** Prefer new tunables under `model:`, `features:`, or `train:` in experiment YAML, validated by **`sparkles/config/schema.py`**.
3. **Thin `train.py`:** Orchestration only—load → build X,y → `build_estimator(cfg)` → fit → save. Estimators and feature assembly live in dedicated modules.
4. **Progress log:** Append a row to **[Progress & change log](#progress--change-log-append-only)** when a phase (or meaningful slice) lands; do not rewrite history.
5. **Re-run rules:** Changing **labels** (barriers, stride, vol lookback, data range, timezone) requires **`label`** (and maybe **`ingest`**) before **`train`**. Changing **only** `model:` / train split usually needs **`train`** only—see **`DEVELOPER.md`**.

---

## Principles

| Principle | Meaning |
|-----------|---------|
| **Contract** | One experiment YAML describes symbol, data, labels, splits, model, and (later) feature flags. |
| **Registry pattern** | `model.type` dispatches to a factory; `features.*` dispatches to named column builders. |
| **Repro bundles** | `model_bundle.joblib` (or successor) should carry estimator, label encoder, **feature column list**, and a **config snapshot** or hash. |
| **Optional deps** | Heavy models (e.g. **xgboost**) stay behind **`[ml]`** in `pyproject.toml` until promoted. |

---

## Roadmap (phased)

### Phase A — Config surface

- Extend **`ModelConfig`**: e.g. `class_weight`, `solver`, `tol` where compatible with sklearn.
- Add **`TrainConfig`** (or `train:` in YAML): min row counts, whether to drop val rows with unseen classes, optional `experiment_name` / `notes` logged to **`experiments.jsonl`**.
- **Done when:** New keys validate; `run_train` reads them; at least one test.

### Phase B — Feature registry

- **`FeatureConfig`** in YAML (booleans or lists) selects builders in **`sparkles/features/`** (e.g. `entry_bar_ohlc`, `label_geometry`).
- Refactor **`build_feature_matrix`** to concatenate selected builders; document each column in **`DEVELOPER.md`**.
- **Done when:** Turning a feature off in YAML changes X without editing `train.py`.

### Phase C — Model registry

- **`sparkles/models/estimators.py`** (or package): `build_estimator(cfg) -> sklearn`-compatible estimator.
- Second implementation: e.g. **`xgboost`** behind **`pip install -e ".[ml]"`** with a clear import error if missing.
- **Done when:** Two `model.type` values work end-to-end; metrics.json records `model.type`.

### Phase D — Preprocessing pipeline

- Optional **`StandardScaler` / `RobustScaler`** in a **`Pipeline`**, fit **only on train** rows.
- **Done when:** Bundle reload applies the same transforms; tests enforce no val fit leakage.

### Phase E — Experiment workflow

- **`sparkles train --dry-run`**: row counts, class balance, feature list.
- Optional **`scripts/`** grid helper writing CSV/JSONL; optional **`configs/experiments/presets/`** copies.
- **Done when:** Documented in **`METHODOLOGY.md`** §9 or **`DEVELOPER.md`**.

### Phase F — Richer evaluation

- Add macro / weighted **F1**, confusion-friendly summaries to **`metrics.json`** and CLI echo.
- **Done when:** Imbalanced `barrier_outcome` is visible in default report output.

---

## Suggested order

**A → B → C → D → E → F** (B before C usually gives better ROI than jumping straight to XGBoost).

---

## Document map

| File | Role |
|------|------|
| **`plan.md`** | Phase 1 iteration approvals and history. |
| **`docs/README.md`** | Index of files under `docs/`. |
| **`docs/ML_EXPANSION.md`** | This file — ML expansion phases and log. |
| **`DEVELOPER.md`** | Where to edit training, features, CLI; links here. |
| **`METHODOLOGY.md`** | Product and data philosophy; terminal examples. |

---

## Progress & change log (append-only)

**Instructions:** Add new entries **only below** this line, newest at the bottom. Do not delete or rewrite prior entries.

| Date (ISO) | Summary | Paths / notes | Phase |
|------------|---------|-----------------|-------|
| 2026-04-11 | Doc created: phased roadmap (A–F), habits, document map; frontmatter todos for tracking. | `docs/ML_EXPANSION.md`, `DEVELOPER.md` pointer, `plan.md` progress row | **Baseline** (doc only) |
| 2026-04-11 | Repo hygiene: **`artifacts/`** + **`.ruff_cache/`** gitignored; **`docs/README.md`** index; **`DEVELOPER.md`** repository layout table; **`scripts/README.md`**. | `.gitignore`, `docs/README.md`, `scripts/README.md`, `DEVELOPER.md`, `README.md`, `docs/ML_EXPANSION.md` | **Docs / layout** |
