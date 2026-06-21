---
name: ML experimentation expansion
overview: "Living plan to grow YAML-driven experiments: more model types, hyperparameters, feature toggles, and preprocessing—without bloating train.py. Complements Phase 1 (plan.md); work here is incremental and owner-approved when treated as a mini-roadmap."
todos:
  - id: config-surface
    content: "Phase A: TrainConfig + extended ModelConfig (class_weight, solver, tol); richer experiments.jsonl"
    status: complete
  - id: feature-registry
    content: "Phase B: FeatureConfig YAML + named builders in features/; refactor dataset join"
    status: complete
  - id: model-registry
    content: "Phase C: Estimator factory + second model (e.g. xgboost under [ml] extra)"
    status: complete
  - id: preprocessing-pipeline
    content: "Phase D: sklearn Pipeline scaler fit-on-train-only; save in bundle"
    status: complete
  - id: experiment-workflow
    content: "Phase E: train --dry-run, optional sweep script, config presets folder"
    status: complete
  - id: evaluation-metrics
    content: "Phase F: macro/weighted F1, imbalance-aware reporting in CLI/metrics.json"
    status: complete
  - id: feature-expansion-g1
    content: "Phase G1: Multi-horizon returns + multi-scale realized vol/range (entry-only)"
    status: complete
  - id: feature-expansion-g2
    content: "Phase G2: Session/time + volume context (rel volume, VWAP distance) ✅"
    status: complete
  - id: feature-expansion-g3
    content: "Phase G3: Bar microstructure + optional market context (SPY/VIX ingest) ✅"
    status: complete
  - id: multi-symbol
    content: "Phase H: Multi-symbol ingest/label/train design (deferred until after Phase I)"
    status: pending
  - id: phase-i-val-backtest
    content: "Phase I1: Val backtest — simulated barrier PnL from predictions + labeled cache ✅"
    status: complete
  - id: phase-i-tp-policy
    content: "Phase I2: TP probability threshold policy (not argmax) ✅"
    status: complete
  - id: phase-i-meta-label
    content: "Phase I3: Meta-labeling spike — binary act-on-entry filter ✅"
    status: complete
  - id: phase-i-sample-weights
    content: "Phase I4: AFML sample weights (label uniqueness) ✅"
    status: complete
  - id: afml-advanced
    content: "Phase I4b–c (optional): purged CV, fractional differentiation"
    status: pending
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

### Phase F — Richer evaluation ✅

- Add macro / weighted **F1**, confusion-friendly summaries to **`metrics.json`** and CLI echo.
- **Done when:** Imbalanced `barrier_outcome` is visible in default report output.

---

## Phase G — Entry-time feature expansion (post–F research backlog)

**Context:** Phase B shipped four YAML toggles and **7 columns** on the RKLB baseline (`log_entry_close`, four **`label_geometry`** fields, **`intraday_range_pct`**, **`log1p_volume`**). A feature review (2026-06-20) concluded this set is **leakage-safe but thin**: it lacks multi-horizon returns, multi-scale vol, session seasonality, and volume *surprises* — the areas most cited in academic momentum/vol literature and practitioner intraday ML.

**Contract (unchanged):** Every new builder uses only information at **`entry_time`** (trailing windows end at the entry bar; daily vol keeps **`shift(1)`** as in **`sparkles/features/volatility.py`**). No `bars_forward`, future OHLC, or post-entry label fields in **`X`**.

**Implementation pattern (each slice):**

1. New boolean(s) or structured block under **`features:`** in **`FeatureConfig`** (`schema.py`).
2. Builder(s) in **`sparkles/features/`** (new module per theme, e.g. `returns.py`, `session.py`).
3. Register in **`registry.py`**; extend **`_required_ohlcv_columns`** / label deps in **`dataset.py`**.
4. Tests in **`tests/test_dataset.py`** (alignment, no NaN explosion on warm-up rows, toggle off → columns absent).
5. Document columns in **`DEVELOPER.md`**; optional preset under **`configs/experiments/presets/`** for A/B vs baseline.

### G1 — Returns and multi-scale volatility (highest ROI) ✅

| YAML flag | Columns | Definition | Notes |
|-----------|---------|------------|-------|
| **`returns_multi_horizon`** | `ret_{5,15,30,60}m` (via **`returns_horizons_bars`**) | Log return from entry close over trailing *k* 1m bars | Implemented 2026-06-20 |
| **`realized_vol_multi`** | `rv_{30,120}m`, `rv_ratio_30_120m` | Std of 1m log returns; optional short/long ratio | **`realized_vol_include_ratio`** |
| **`range_vol_multi`** | `parkinson_30m`, `atr_norm_30m` | Parkinson range vol + normalized ATR | **`range_vol_window_bars`**, **`range_vol_include_atr_norm`** |

**Preset:** **`configs/experiments/presets/g1_features_v1.yaml`** (G1 + champion XGB hyperparams).

**Warm-up:** Rows with position `< max(horizons, windows)` dropped (default **120** bars). Logged at INFO in **`build_feature_matrix`**.

**Done when:** ✅ Flags validate; preset dry-runs; tests in **`tests/test_intraday_features.py`**.

### G2 — Session time and volume context ✅

| YAML flag | Columns | Definition | Notes |
|-----------|---------|------------|-------|
| **`session_time`** | `minutes_since_open`, `minutes_to_close`, `sin_time`, `cos_time` | Exchange-TZ session progress; cyclical time | Implemented 2026-06-21 |
| **`volume_context`** | `rel_volume`, `log_rel_volume` | Entry volume / rolling median volume (trailing, entry-only) | **`volume_median_window_bars`** (default 60) |
| **`vwap_distance`** | `vwap_session_dist_pct` | `(close − session_VWAP) / session_VWAP` through entry bar | Session VWAP from typical price × volume |

**Preset:** **`configs/experiments/presets/rklb_daytrade_g1_g2_v1.yaml`** (G1 + G2 on day-trade v2 base).

**Warm-up:** G2 adds **`volume_median_window_bars`** when **`volume_context`** is on; combined with G1 via **`feature_warmup_bars()`** in **`build_feature_matrix`**.

**Done when:** ✅ Session boundaries use **`exchange_timezone`**; tests in **`tests/test_session_features.py`**.

### G3 — Bar microstructure and optional market regime ✅

| YAML flag | Columns | Definition | Notes |
|-----------|---------|------------|-------|
| **`bar_microstructure`** | `close_loc_value`, `bar_body_pct` | `(C−L)/(H−L)`, `(C−O)/C` on entry bar | Implemented 2026-06-21 |
| **`market_context`** | `spy_ret_{N}m`, `vix_chg_1d` | Trailing SPY 1m log return; prior-session daily pct change of **volatility proxy** | **`market_spy_return_bars`** (default 15); requires **`context_ingest`**. TwelveData has **no spot VIX/^VIX** — use **VIXY** (or VXX) **`1day`** + **`twelvedata_exchange: CBOE`**. Feature column name stays `vix_chg_1d`. |

**Context ingest (YAML):** list SPY/VIX under **`context_ingest.symbols`** with the **same `data_start`/`data_end`** as the main symbol. **`sparkles ingest`** downloads RKLB plus context symbols (respects cache TTL; use **`--force`** only when needed).

**Cache paths:** `data/cache/SPY_1min_{start}_{end}.parquet`, `data/cache/VIXY_1day_{start}_{end}.parquet` (VIXY + CBOE exchange — not spot VIX). Download: **`sparkles ingest -c … -s SPY -i 1min`**, **`-s VIXY -i 1day`**.

**Preset:** **`configs/experiments/presets/rklb_daytrade_g1_g2_g3_v1.yaml`**.

**Done when:** ✅ Microstructure needs no extra vendor data; **`market_context`** documents credit cost, cache paths, and validates SPY+VIX in YAML.

### Phase G — evaluation habit

After each G slice lands, compare to **`xgb_d3_reg_v1`** preset on **the same** RKLB train/val dates:

- Primary: **`val_f1_macro`** (imbalanced barriers).
- Secondary: per-class val F1 (`take_profit` vs `stop_loss`), train–val gap (overfit).
- Log enabled **`features`** snapshot in **`experiments.jsonl`** (already supported).

**References (starting bibliography):** López de Prado, *Advances in Financial Machine Learning* (2018); Jegadeesh & Titman (1993); Andersen et al. (2001) realized vol; Parkinson (1980); Amihud (2002) illiquidity; Easley et al. (2012) microstructure.

---

### Phase H — Multi-symbol experiments (after G stabilizes on one ticker)

**Not started.** Today each experiment YAML has a single **`symbol`**; training loads one labeled Parquet. Expanding to multiple tickers is a **product/design** choice, not just more ingest.

**Recommended order:** **Expand features on RKLB first (Phase G), then add symbols** — see [When to add more symbols](#when-to-add-more-symbols-features-first) below.

**Design options (pick one when implementing):**

| Approach | Pros | Cons |
|----------|------|------|
| **Separate model per symbol** | Simple; barrier geometry per name; no cross-ticker leakage | More trains; no shared statistical strength |
| **Pooled train, symbol indicator** | More rows; learns generic intraday patterns | Needs **`symbol`** categorical or shared normalization; barrier/outcome mix differs by name |
| **Shared features, per-symbol val** | Generalization test | Requires multi-symbol ingest/label pipeline and reporting |

**Deliverables (draft):**

- Documented ingest/label batch for a **small universe** (e.g. 3–5 liquid names similar to RKLB).
- YAML or script pattern for pooled vs per-symbol runs without duplicating barrier math.
- **`experiments.jsonl`** / CSV columns: **`symbol`** (already present) + optional **`universe_id`**.
- Credit-aware batch ingest (existing throttling; no new polling loops).

**Done when:** Owner picks an approach; at least two symbols train and val with frozen feature list; comparison doc in **`METHODOLOGY.md`** or **`DEVELOPER.md`**.

---

### Phase I — Validation economics and tradable policy (post–G; before H)

**Context:** **G1–G3 are complete** on RKLB day-trade v2. The current champion preset is **`configs/experiments/presets/rklb_daytrade_champion_v1.yaml`** (reproduces **`Trial_RB_G1_G2_G3_v1`**, run **`20260621T161419_221266Z`**: val macro F1 **~0.525**, best **balanced** take-profit F1 among full G1+G2+G3 trials). Classification metrics answer “did we predict the label?” — not “would this policy trade profitably?” **Phase I** closes that gap **before** multi-symbol (**H**) or live / Robinhood paths.

**Repo state (2026-06-21):** **`sparkles backtest`** simulates val policy PnL from **`predictions.parquet`** + labeled cache → **`backtest_summary.json`** + **`backtest_trades.parquet`**. Threshold sweep (**I2**) and meta-label trainer (**I3**) are **not** implemented yet.

**Non-negotiable workflow (same as `plan.md`):**

1. **Do not start the next I-slice** until the owner explicitly approves in chat (e.g. “approved — continue to I2”) or updates this doc / **`plan.md`**.
2. **Append** the [Progress & change log](#progress--change-log-append-only) when a slice lands.
3. **No TwelveData polling loops** — backtests read **cached** labeled + ingest Parquet and exported predictions only.

**Champion baseline for all I slices:** train with **`rklb_daytrade_champion_v1.yaml`** on **`rklb_daytrade_v2.yaml`**; compare new tooling against **`metrics.json`** *and* the new economics metrics (I1+).

---

#### I1 — Val backtest from labeled outcomes ✅

| | |
|--|--|
| **Goal** | Turn val **`predictions.parquet`** + labeled cache into a **simple simulated long-only policy report**: entries taken, outcome mix, gross return proxy per trade, cumulative val curve, optional day-trade ledger stats. |
| **Policy v1** | **`y_pred == take_profit`** (argmax). **PnL:** TP → **`+tp_move_effective`**, SL → **`-sl_move`**, vertical/end_of_data → OHLCV close at **`bars_forward`**. |
| **CLI** | **`sparkles backtest -c configs/experiments/rklb_daytrade_v2.yaml --run <run_id>`** (default: latest run with predictions). Flags: **`--split val`**, **`--no-day-trade-cap`**. |
| **Artifacts** | **`backtest_summary.json`**, **`backtest_trades.parquet`** next to **`metrics.json`**. Module: **`sparkles/backtest/`**; tests **`tests/test_val_backtest.py`**. |
| **Champion sanity (2026-06-21)** | Run **`20260621T161419_221266Z`**: 133 val signals → **70 taken** (63 blocked by 3-in-5 day-trade cap on same-session exits); hit rate TP on taken **4.3%**; gross return sum **+0.90** (fractional, overlapping entries, no fees). |
| **Done when** | ✅ Owner can run backtest on champion without API calls; assumptions in **`DEVELOPER.md`**. |
| **Owner approval to proceed to I2** | `[ ]` Date: ___________ |

---

#### I2 — TP probability threshold policy ✅

| | |
|--|--|
| **Goal** | Replace **argmax** entry rule with **`proba_take_profit >= threshold`**; sweep thresholds on **val only** and report precision/recall **and** backtest economics from I1. |
| **CLI** | **`sparkles backtest --threshold 0.35`** — single threshold run. **`sparkles backtest --sweep`** — grid (default step **0.05**) → **`backtest_threshold_sweep.csv`** + **`.json`** with **`suggested_threshold`**. |
| **YAML (opt-in)** | **`train.entry_threshold_take_profit`** — when set, **`sparkles backtest`** (no **`--threshold`**) uses threshold policy; **train fit unchanged** (argmax labels only). |
| **Champion sweep (2026-06-21, run `20260621T161419_221266Z`)** | Argmax (I1): 133 signals, **4.3%** hit on taken. Threshold **0.35**: 190 signals, **12.1%** precision, **29%** recall, gross sum **+1.48** (100 taken). Threshold **0.50**: 63 signals, **15.9%** precision. Mechanical max-precision suggestion at **0.65** (6 signals, 50% precision) — too sparse for production; **practical knee ~0.35–0.50**. |
| **Done when** | ✅ Sweep on champion val without API calls; default backtest remains argmax unless **`--threshold`** or YAML opt-in. |
| **Owner approval to proceed to I3** | `[ ]` Date: ___________ |

---

#### I3 — Meta-labeling spike ✅

| | |
|--|--|
| **Goal** | AFML-style **secondary binary model**: “**act on this entry?**” on top of primary **`proba_take_profit`** gate. |
| **CLI** | **`sparkles meta-label train -c … --run <primary_id>`** → **`meta_label_bundle.joblib`** + **`meta_label_metrics.json`**. **`sparkles meta-label compare …`** → **`meta_label_compare.json`** (argmax vs threshold vs meta on val). Uses frozen **`experiment_config.json`** from the primary run when present. |
| **Train contract** | Meta fit rows = **train split only**, primary-gated (`proba_take_profit >= primary_threshold`, default **0.35**). Target = **`y_true == take_profit`**. Features = primary **`proba_*`**, **`max_proba`**. |
| **Champion compare (2026-06-21, τ=0.35, meta_act=0.5)** | **argmax:** 133 signals, 13.5% precision, gross **+0.90**. **threshold:** 190 signals, 12.1% precision, gross **+1.48**. **meta:** 101 signals, **16.8%** precision, gross **+1.14** — higher precision, fewer false entries vs threshold-only. |
| **Done when** | ✅ End-to-end val comparison on champion; documented in **`DEVELOPER.md`**. Train **`run_train`** unchanged. |
| **Owner approval to proceed to I4+ or H** | `[ ]` Date: ___________ |

---

#### I4 — Sample weights (label uniqueness) ✅

| | |
|--|--|
| **Goal** | Down-weight **overlapping** triple-barrier labels when **`label_entry_stride` < barrier horizon** (AFML avg inverse concurrency using **`bars_forward`**). |
| **YAML (opt-in)** | **`train.sample_weight_method: none \| uniqueness`** — default **`none`** preserves champion / prior runs. Combines with **`model.class_weight`** when both set (product per row). |
| **Implementation** | **`sparkles/models/sample_weights.py`**; wired in **`run_train`** → **`metrics.json`** records **`sample_weight_method`**, **`sample_weight_mean`**, **`sample_weight_min`**. |
| **When to use** | Strided day-trade labels (e.g. stride **10**) where many entries share overlapping forward paths; compare val F1 / backtest before promoting. |
| **Done when** | ✅ Opt-in YAML + tests **`test_sample_weights.py`**; default unchanged. |

#### I4b–c — Optional backlog (**awaiting owner approval**)

| Item | Purpose |
|------|---------|
| **Purged / embargo CV** | Reduce leakage across overlapping label windows in **`run_trials.py`** / notebook sweeps |
| **Fractional differentiation** | Stationary features that retain memory (López de Prado); alternative to raw **`log_entry_close`** |

---

**Phase I — evaluation habit**

After **I1**, every model trial report should include **both**:

- Classification: **`val_f1_macro`**, per-class val F1 (especially **`take_profit`**).
- Economics: val **entries**, **hit rate on acted entries**, **gross PnL proxy**, simple **max drawdown** (definitions fixed in I1 doc).

Do **not** tune hyperparameters on val economics until I1 exists — otherwise val becomes a second training set.

---

## When to add more symbols (features first)

**Default recommendation: add and validate Phase G features on one symbol (RKLB) *before* scaling to multiple tickers.**

| Do features first (RKLB) | Add symbols later |
|--------------------------|-------------------|
| Easier to interpret val F1 moves (one barrier mix, one price history) | Multi-symbol multiplies **ingest + label** time and **TwelveData credits** |
| Current code path is **one symbol per YAML** | Pooled training needs extra design (normalization, symbol feature, per-symbol val) |
| **`log_entry_close`**-style level features may be symbol-specific; returns/vol ratios generalize better once proven | Class balance and vol scaling differ by ticker — confounds feature A/B if changed together |
| Champion **`xgb_d3_reg_v1`** is RKLB-specific baseline | After G, add **1–2 similar names** as a **generalization check**, not as the first lever |

**Exception:** After G1, optionally add **one** second symbol only to sanity-check that new features aren’t RKLB-overfit — but keep the main feature iteration loop on a single ticker until val metrics stabilize.

---

## Suggested order

**Completed:** A → B → C → D → E → F

**Completed:** **G1**, **G2**, **G3** on RKLB day-trade v2.

**Next (owner-approved slices):** **Phase I** (**I1** val backtest → **I2** TP threshold policy → **I3** meta-label spike) → optional **I4+** (sample weights, purged CV) → **Phase H** (multi-symbol) only after I stabilizes tradable metrics on one ticker.

B before C gave better ROI than jumping to XGBoost; **G before H** and **I before H** follow the same logic: prove features and **policy economics** on one symbol before scaling universe or going live.

---

## Current feature baseline (Phase B recap)

For reviewers — what ships today when all **`features.*`** flags are **true** in **`rklb_baseline.yaml`**:

| Group | Columns |
|-------|---------|
| **`log_entry_close`** | `log_entry_close` |
| **`label_geometry`** | `sigma_ann_at_entry`, `vol_scale_ratio`, `tp_move_effective`, `sl_move` |
| **`intraday_range_pct`** | `intraday_range_pct` |
| **`log1p_volume`** | `log1p_volume` |

**Caution:** **`label_geometry`** encodes barrier *setup* for that entry (tied to YAML barriers). Keep it, but Phase G adds **orthogonal market-state** columns so the model is not only learning label construction.

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
| 2026-04-10 | **Phase A complete:** **`TrainConfig`** (`train:` YAML): `min_train_rows`, `min_val_rows`, `drop_val_unseen_classes`, `experiment_name`, `notes`. **`ModelConfig`**: `solver`, `tol`, `class_weight` (null / balanced / per-outcome map → sklearn ints in `run_train`). Richer **`experiments.jsonl`** fields. Tests in **`tests/test_train_smoke.py`**. **`DEVELOPER.md`** + commented examples in **`rklb_baseline.yaml`**. | `sparkles/config/schema.py`, `sparkles/models/train.py`, `sparkles/config/__init__.py`, `tests/test_train_smoke.py`, `DEVELOPER.md`, `configs/experiments/rklb_baseline.yaml`, `docs/ML_EXPANSION.md` | **Phase A** |
| 2026-04-10 | **Phase B complete:** **`FeatureConfig`** + **`features:`** YAML; **`sparkles/features/builders.py`** (`EntryFeatureContext` + column builders); **`sparkles/features/registry.py`** (`assemble_feature_columns`). **`build_feature_matrix`** selects groups and required OHLCV columns from config. **`metrics.json`** / **`experiments.jsonl`** include **`features`** snapshot. Tests **`tests/test_dataset.py`**, **`tests/test_schema_features.py`**. | `sparkles/config/schema.py`, `sparkles/features/*.py`, `sparkles/models/train.py`, `sparkles/config/__init__.py`, `tests/`, `DEVELOPER.md`, `configs/experiments/rklb_baseline.yaml`, `docs/ML_EXPANSION.md` | **Phase B** |
| 2026-04-10 | **Phase C complete:** **`sparkles/models/estimators.py`** — `build_estimator`, `resolve_logistic_class_weight`, `xgboost_fit_sample_weight`. **`ModelKind`** + **`xgboost_classifier`** behind **`[ml]`**; **`metrics.json`** **`model_type`**; tests **`tests/test_estimators.py`**, optional XGB end-to-end in **`tests/test_train_smoke.py`**. | `sparkles/models/estimators.py`, `sparkles/models/train.py`, `sparkles/models/__init__.py`, `sparkles/config/schema.py`, `sparkles/reporting/summary.py`, `sparkles/cli.py`, `tests/`, `DEVELOPER.md`, `README.md`, `configs/experiments/rklb_baseline.yaml`, `docs/ML_EXPANSION.md` | **Phase C** |
| 2026-06-20 | **Phase E complete (owner approved):** **`sparkles train --dry-run`**; **`prepare_training_data`** / **`dry_run_train`**; **`load_experiment_config_merged`**; **`configs/experiments/presets/`**; **`scripts/run_trials.py`**; tests **`test_config_merge.py`**, dry-run in **`test_train_smoke.py`**; **`DEVELOPER.md`**, **`METHODOLOGY.md`**, **`scripts/README.md`**. | `sparkles/models/train.py`, `sparkles/config/load.py`, `sparkles/cli.py`, `scripts/run_trials.py`, `configs/experiments/presets/`, `tests/`, docs | **Phase E** |
| 2026-06-20 | **Phase F complete (owner approved):** **`sparkles/models/evaluation.py`** — macro/weighted F1 in **`metrics.json`**, **`experiments.jsonl`**, CLI train echo, **`sparkles report`** per-class val summary; CSV column priority; tests **`test_evaluation.py`**, updated reporting/train smoke. | `sparkles/models/evaluation.py`, `sparkles/models/train.py`, `sparkles/reporting/summary.py`, `sparkles/tracking/experiments_csv.py`, `sparkles/cli.py`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase F** |
| 2026-06-20 | **Phase D complete (owner approved):** **`preprocess.scaler`** YAML (`none` \| `standard` \| `robust`); train-only sklearn **Pipeline**; bundle **`preprocess_scaler`** + **`predict_from_bundle`**; tests **`test_preprocess.py`**. | `sparkles/config/schema.py`, `sparkles/models/preprocess.py`, `sparkles/models/train.py`, `sparkles/reporting/summary.py`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase D** |
| 2026-06-20 | **Feature review + roadmap Phases G–I:** entry-time expansion backlog (returns, multi-scale vol, session/volume, microstructure, optional market context); multi-symbol Phase H; AFML Phase I; **features-before-symbols** guidance. Doc only — no code. | `docs/ML_EXPANSION.md`, `plan.md` | **Phase G planning** |
| 2026-06-20 | **Phase G1 complete (owner approved):** `returns_multi_horizon`, `realized_vol_multi`, `range_vol_multi` YAML + builders; full-OHLCV trailing windows; warm-up row drop; preset **`g1_features_v1.yaml`**; tests **`test_intraday_features.py`**. | `sparkles/config/schema.py`, `sparkles/features/intraday.py`, `sparkles/features/builders.py`, `sparkles/features/dataset.py`, `sparkles/features/registry.py`, `configs/experiments/presets/g1_features_v1.yaml`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase G1** |
| 2026-06-21 | **Phase G2 complete (owner approved):** `session_time`, `volume_context`, `vwap_distance` YAML + builders in **`session.py`**; preset **`rklb_daytrade_g1_g2_v1.yaml`**; tests **`test_session_features.py`**. | `sparkles/features/session.py`, `sparkles/config/schema.py`, `sparkles/features/registry.py`, `sparkles/features/dataset.py`, `configs/experiments/presets/rklb_daytrade_g1_g2_v1.yaml`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase G2** |
| 2026-06-21 | **Phase G3 complete (owner approved):** `bar_microstructure`, `market_context`; **`context_ingest`** SPY/VIX download; preset **`rklb_daytrade_g1_g2_g3_v1.yaml`**. | `sparkles/features/microstructure.py`, `sparkles/features/market_context.py`, `sparkles/data/context_ingest.py`, `sparkles/config/schema.py`, `configs/experiments/rklb_daytrade_v2.yaml`, `tests/test_g3_features.py`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase G3** |
| 2026-06-21 | **Champion preset + Phase I plan:** **`rklb_daytrade_champion_v1.yaml`** (G1+G2+G3 + Trial_RB_G1_G2_G3_v1 hparams); Phase I split into **I1** val backtest, **I2** TP threshold policy, **I3** meta-label spike, **I4+** AFML backlog; H deferred until after I. | `configs/experiments/presets/rklb_daytrade_champion_v1.yaml`, `docs/ML_EXPANSION.md`, `DEVELOPER.md`, `plan.md` | **Phase I planning** — **awaiting owner approval for I1** |
| 2026-06-21 | **Phase I1 complete (owner approved):** **`sparkles backtest`** — val policy PnL from **`predictions.parquet`** + labeled cache; **`backtest_summary.json`** / **`backtest_trades.parquet`**; day-trade ledger optional; tests **`test_val_backtest.py`**. | `sparkles/backtest/`, `sparkles/cli.py`, `tests/test_val_backtest.py`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase I1** — **blocked until owner approves I2** |
| 2026-06-21 | **Phase I2 complete (owner approved):** TP threshold policy (**`--threshold`**, **`--sweep`**); **`train.entry_threshold_take_profit`** YAML opt-in; **`backtest_threshold_sweep.csv/json`**; tests **`test_threshold_sweep.py`**. Champion knee ~**0.35–0.50** precision. | `sparkles/backtest/threshold_sweep.py`, `sparkles/config/schema.py`, `sparkles/cli.py`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase I2** — **blocked until owner approves I3** |
| 2026-06-21 | **Phase I3 complete (owner approved):** **`sparkles meta-label train/compare`**; binary filter on primary-gated signals; **`meta_label_*` artifacts**; tests **`test_meta_label.py`**. Champion: meta **16.8%** precision vs threshold **12.1%** at τ=0.35. | `sparkles/backtest/meta_label.py`, `sparkles/cli.py`, `sparkles/config/schema.py`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase I3** — **blocked until owner approves I4+ or H** |
| 2026-06-21 | **Phase I4 complete (owner approved):** **`train.sample_weight_method: uniqueness`** — AFML label-uniqueness weights at fit; opt-in; tests **`test_sample_weights.py`**. | `sparkles/models/sample_weights.py`, `sparkles/models/train.py`, `sparkles/config/schema.py`, `tests/`, `DEVELOPER.md`, `docs/ML_EXPANSION.md`, `plan.md` | **Phase I4** — **blocked until owner approves I4b/c or H** |
