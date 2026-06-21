# Sparkles developer map

Quick reference for **where to change what** in Phase 1. Conceptual methodology: **[METHODOLOGY.md](METHODOLOGY.md)**. Roadmap and approvals: **`plan.md`**. GitHub overview: **[README.md](README.md)**. **ML expansion** (models, features, YAML roadmap): **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)**.

## Repository layout

| Path | Role |
|------|------|
| **`sparkles/`** | Package: `config`, `data`, `features`, `journal`, `labels`, `models`, `reporting`, `risk`, `tracking` (`experiments.py`, `experiments_csv.py`), `cli.py`. |
| **`configs/experiments/`** | Experiment YAML (e.g. `rklb_baseline.yaml`); **`presets/`** overlays for batch trials. |
| **`tests/`** | `pytest` only; mirror modules under `sparkles/` where practical. |
| **`scripts/`** | Non-installed helpers (`quick_try_vol.py`, **`run_trials.py`**); run with `python scripts/...`. |
| **`docs/`** | Longer-lived docs beyond root summaries (`ML_EXPANSION.md`, this layout index in `docs/README.md`). |
| **`data/cache/`** | Parquet from **`ingest`** / **`label`** (gitignored). |
| **`data/journal/`** | Optional personal trade CSVs (see README there; `*.csv` gitignored). |
| **`artifacts/`** | Training runs + `experiments.jsonl` from **`train`** (gitignored). |
| **Root** | `README.md`, `METHODOLOGY.md`, `DEVELOPER.md`, `plan.md`, `pyproject.toml`, `.env.example`, **`Sparkles.code-workspace`** (optional VS Code multi-root). |
| **`.cursor/rules/`** | Cursor agent rules (API credits, iterative plan). |

## Practice symbol (ticker)

- **File:** `configs/experiments/rklb_baseline.yaml` (or your own copy under `configs/experiments/`)
- **Day-trade labels (v1):** `configs/experiments/rklb_daytrade_v1.yaml` — 15% TP / 10% SL / 12% min profit / **`vertical_max_trading_days: 1`** / **`label_entry_stride: 15`**. Same ingest Parquet as baseline; **must re-label** → `…_labeled_…_s15.parquet`. Metrics **not comparable** to baseline champion. Optional G1 preset: **`presets/rklb_daytrade_g1_v1.yaml`**.
- **Day-trade labels (v2):** `configs/experiments/rklb_daytrade_v2.yaml` — **3% TP / 5% SL / 1.5% min profit** / 1-day vertical / stride 15; labeled cache **`…_s15_dt_v2.parquet`** via **`label_cache_suffix`**. Preset **`presets/rklb_daytrade_v2_g1.yaml`**.
- **Field:** `symbol` (default **RKLB**)
- **CLI:** pass `--config path/to/experiment.yaml` to `sparkles` subcommands

## Training code (Python you edit often)

- **File:** `sparkles/models/train.py` — **`run_train(cfg)`** loads labeled + ingest Parquets, **`build_feature_matrix`**, time-splits, fits **`model.type`** via **`sparkles/models/estimators.py`**, writes **`model_bundle.joblib`**, **`metrics.json`**, **`experiment_config.json`** (full YAML-equivalent snapshot), and (by default) **`predictions.parquet`** per **`train.export_predictions`**. Appends **`experiments.jsonl`**. CSV export: **`sparkles experiments export -c …`**
- **Sample weights (I4):** **`train.sample_weight_method: none | uniqueness`** — **`uniqueness`** applies AFML avg inverse concurrency from **`bars_forward`** at fit (useful when **`label_entry_stride`** creates overlapping labels). Default **`none`**. Combines with **`model.class_weight`** when both set.
- **Estimators:** `sparkles/models/estimators.py` — **`build_estimator`**: `logistic_regression` (sklearn, core deps) or **`xgboost_classifier`** (install **`pip install -e ".[ml]"`**). XGBoost hyperparameters: **`model.xgb_n_estimators`**, **`xgb_max_depth`**, **`xgb_learning_rate`**, **`xgb_subsample`**, **`xgb_colsample_bytree`**, **`xgb_min_child_weight`**. YAML **`model.class_weight`** maps to sklearn for logistic regression and to **`sample_weight`** for XGBoost when not null.
- **Preprocessing (Phase D):** **`preprocess.scaler`** in YAML — **`none`** (default), **`standard`**, or **`robust`**. Fitted **only on train** inside a sklearn **Pipeline** saved in **`model_bundle.joblib`** as **`estimator`**; bundle also stores **`preprocess_scaler`**. Reload helpers: **`sparkles/models/preprocess.py`** (`load_model_bundle`, `validate_bundle_preprocess`, `predict_from_bundle`). Trees (XGBoost) often need **`none`**; logistic may benefit from scaling.
- **Registry:** `sparkles/models/registry.py` — `new_run_id`, `run_artifact_dir`, `save_bundle`, `save_json`.
- **Features at entry only:** Controlled by **`features:`** in YAML (`FeatureConfig` in `schema.py`). Each flag includes a builder group from **`sparkles/features/registry.py`** (see **`sparkles/features/builders.py`**):
  - **`log_entry_close`:** column `log_entry_close` — `log(entry_close)` from the label row.
  - **`label_geometry`:** `sigma_ann_at_entry`, `vol_scale_ratio`, `tp_move_effective`, `sl_move` — barrier/vol snapshot from labeling.
  - **`intraday_range_pct`:** `(high − low) / entry_close` on the **entry bar** from ingest OHLCV (requires `high`, `low`, `close` on OHLCV).
  - **`log1p_volume`:** `log1p(volume)` on the entry bar (volume optional on OHLCV; missing → zeros).
  - **`returns_multi_horizon` (G1):** `ret_{N}m` — log return over trailing *N* 1m bars ending at entry (default horizons 5, 15, 30, 60).
  - **`realized_vol_multi` (G1):** `rv_{N}m` — std of 1m log returns over trailing window; optional `rv_ratio_{short}_{long}m`.
  - **`range_vol_multi` (G1):** `parkinson_{N}m`, optional `atr_norm_{N}m` — range vol and normalized ATR (default window 30).
  G1 groups read the **full ingest OHLCV** up to each entry bar (no lookahead). Rows before the warm-up window are dropped automatically (max horizon/window, default 120 bars).
  All default **true** (Phase 1 column set). No future path / `bars_forward` in the feature matrix.
- **Config:** `model.*`, `train.*` (includes **`export_predictions`**), and **`features.*`**. Optional **`journal:`** — **`csv_path`**: your trade log (repo-relative or absolute). Optional **`live_ingest:`** — Phase 2 Plan A near-live refresh (**`enabled`** default **false**); see **`docs/plan-phase2-01-data-ingest.md`**. **CSV:** date column **`entry_date`**, **`date`**, **`open_date`**, or **`entry`** (ISO). Optional **`symbol`** / **`ticker`** filters rows to the experiment **`symbol`**. Extra columns (`exit_date`, `shares`, `pnl_pct`, `notes`, …) are kept in the merge output. **Long holds:** one row per **open** is fine; compare aligns on **entry session date** to aggregated model predictions that day (not daily P&L over the hold).
- **Journal compare:** `sparkles journal compare -c …` — loads **`journal.csv_path`**, latest run with **`predictions.parquet`** (or **`--run <run_id>`**), aggregates predictions by **`session_date`** for **`--split val`** (default), **`train`**, or **`both`**, left-joins journal **`entry_date`**, writes **`journal_compare.csv`** in the run folder. Template: **`configs/examples/journal_trades.example.csv`**.
- **Val backtest / policy (Phase I1–I3):** Post-train CLI on the run folder (not in the notebook yet):
  - **`sparkles backtest -c … --run <id>`** — argmax; **`--threshold`** / **`--sweep`** (I2)
  - **`sparkles meta-label train …`** then **`sparkles meta-label compare …`** (I3)
  - Primary **`sparkles train`** / **`run_train`** unchanged — I1–I3 are policy layers on predictions + bundles.
- **Prerequisites:** `sparkles ingest` then `sparkles label` for the same `symbol`, `data_start`, `data_end`, and `label_entry_stride` as in the YAML.

## Hyperparameter trials (ML expansion Phase E)

- **Dry-run before train:** `sparkles train -c configs/experiments/rklb_baseline.yaml --dry-run` — prints train/val row counts, **class balance**, enabled **features**, and feature column names; exits **1** if ingest/label missing or split floors fail.
- **Preset overlays:** `configs/experiments/presets/*.yaml` — **`xgb_d3_reg_v1.yaml`** is the RKLB **baseline-label** champion (2026-06-20); **`rklb_daytrade_champion_v1.yaml`** is the **day-trade v2 + G1+G2+G3** champion (2026-06-21); **`rklb_daytrade_champion_uniqueness_v1.yaml`** adds **`train.sample_weight_method: uniqueness`**. Merge with **`load_experiment_config_merged(base, preset)`** or **`python scripts/run_trials.py --preset …`**
- **Batch trials:** `python scripts/run_trials.py` (default base: **`rklb_baseline.yaml`**, all presets). Flags: **`--dry-run`**, **`--preset path/to/overlay.yaml`**, **`--no-export`**, **`-o artifacts/training_log.csv`**. After real trains, refreshes the wide CSV for spreadsheet comparison.
- **Grid search:** `python scripts/run_grid_search.py --grid configs/experiments/grids/rklb_daytrade_xgb_v1.yaml` — cartesian sweep over dotted YAML paths (`model.*`, `features.*`, `train.*`). Spec files live under **`configs/experiments/grids/`**; results CSV under **`artifacts/grid_search/`**. Use **`fixed.train.export_predictions: none`** in the spec to skip Parquet export during large grids.
- **Notebook console:** **`notebooks/sparkles_train_console.ipynb`** — set **`RUN_MODE`** to **`"single"`** or **`"grid"`**; inline **`GRID_SPEC`** or **`GRID_FROM_YAML`** for sweeps (`pip install -e ".[notebook]"`).
- **Compare results:** **`sparkles experiments export -c …`** or open **`artifacts/training_log.csv`**; sort by **`val_f1_macro`** (preferred for imbalanced labels) or **`val_accuracy`**.
- **Feature expansion:** Phases **G1–G3** and **I1–I4** complete. **Next:** optional **I4b/c** (purged CV, fractional diff) or **Phase H** per **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)**.

## Labeling and minimum profit per trade

- **Config:** `min_profit_per_trade_pct` in experiment YAML (fraction, e.g. `0.02` = **2%**)
- **Implementation:** `sparkles/labels/triple_barrier.py` — vol-scaled TP/SL (ratio `sigma_t / sigma_ref` clamped by `barrier_vol_scale_min` / `barrier_vol_scale_max`), then **effective TP move** `max(min_profit_per_trade_pct, tp_move_from_vol)`. Forward scan on 1m `high`/`low`; on a tie in the same bar, **stop** is checked before **take-profit** (pessimistic long). Vertical exit uses **trading-day** count from entry (`vertical_max_trading_days`). **`label_entry_stride`** in YAML sets how many 1m bars between candidate entries (**`configs/experiments/rklb_baseline.yaml`** documents **`390`** vs **`1`** trade-offs).
- **CLI:** `sparkles label -c configs/experiments/rklb_baseline.yaml` loads the ingest Parquet for `symbol` + `data_start`/`data_end`, adds vol if missing, writes labeled Parquet, prints `barrier_outcome` value counts. `-v` enables progress logging.
- **On disk (labeled):** `{SYMBOL}_labeled_{data_start}_{data_end}_s{label_entry_stride}.parquet` under `paths.cache_dir`.

## Barriers, vol lookback, vertical horizon

- **Config:** `profit_barrier_base`, `stop_loss_base`, `vol_lookback_trading_days`, `vertical_max_trading_days`, optional `barrier_vol_scale_min`, `barrier_vol_scale_max`, `label_entry_stride` in experiment YAML
- **Volatility (Iteration 3):** `sparkles/features/volatility.py`
  - **No lookahead:** all 1m bars on session date *D* get the same estimate: rolling std of **daily** log returns over `vol_lookback_trading_days`, then **`shift(1)`** so *D*’s own closing print is **not** in the window.
  - **Outputs:** `sigma_daily_{N}d` (1-day units) and `vol_{N}d_ann` (= daily × √252) added by `add_volatility_columns` / `add_volatility_from_config`.
  - **Usage:** load Parquet from ingest → `add_volatility_from_config(df, cfg)` (expects `close` column and bar index).

**Quick script (from repo root, after ingest):**

```bash
python scripts/quick_try_vol.py
python scripts/quick_try_vol.py -c configs/experiments/rklb_baseline.yaml
```

**Or in a Python REPL / notebook** (not raw PowerShell — use `python` first):

```python
import pandas as pd
from pathlib import Path
from sparkles.config import load_experiment_config
from sparkles.features import add_volatility_from_config

cfg = load_experiment_config(Path("configs/experiments/rklb_baseline.yaml"))
df = pd.read_parquet("data/cache/RKLB_1min_2024-01-01_2024-12-31.parquet")
df2 = add_volatility_from_config(df, cfg)
```

## Day-trade cap (3 in 5 US business days)

- **Implementation:** `sparkles/risk/day_trade_ledger.py` — **`DayTradeLedger`** records one **day-trade event** per same-day round trip (two closes same session date → two records). **`rolling_us_business_days_ending(as_of, periods)`** builds the window: last **`rolling_business_days`** **weekdays** (Mon–Fri) ending at **`as_of`** (weekends roll back to Friday). **NYSE holidays are not skipped** in v1; upgrade to a market calendar if you need exact sessions.
- **API:** **`count_in_window(as_of)`**, **`can_add_day_trade(session_date)`**, **`record_if_allowed(session_date)`** (returns whether recorded). Use **`record()`** only when you intentionally bypass the cap (e.g. tests).
- **Config:** `max_day_trades` (default **3**), `rolling_business_days` (default **5**) in experiment YAML.
- **CLI dry-run:** `sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml` prints counts for **today** (local date). Optional: `--as-of 2026-04-01` and `--history 2026-03-25,2026-03-26,2026-03-26` (comma-separated ISO dates; repeats count as separate events).

## Data paths and API key

- **Cache:** `paths.cache_dir` in YAML (default `data/cache`)
- **Artifacts:** `paths.artifacts_dir` (default `artifacts`)
- **Secrets:** `.env.example` → copy to `.env`, set `TWELVEDATA_API_KEY`

### Historical ingest (Iteration 2)

- **CLI:** `sparkles ingest -c configs/experiments/rklb_baseline.yaml` downloads the **experiment symbol** at **1min** (default). Use **`--symbol` / `-s`** and **`--interval` / `-i`** for each cache file independently (same `data_start`/`data_end` from YAML):
  ```bash
  sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml              # RKLB 1min
  sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s SPY -i 1min
  sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s VIXY -i 1day
  ```
  A fresh RKLB cache does **not** block SPY/VIX downloads (or vice versa).
- **Flags:** `--force` / `-f` bypasses `cache_ttl_hours` and re-downloads the full `data_start`–`data_end` range for **that symbol only**; `-v` enables verbose chunk logging.
- **Config:** `ingest_chunk_calendar_days` (default **10** — fewer HTTP calls); `ingest_sleep_seconds_between_chunks` (default **20** s — stay under free-tier **~8 API credits/minute**); `twelvedata_per_minute_credit_wait_seconds` (default **65** — wait for the next minute when TwelveData returns a per-minute credit error, instead of fast retries); `twelvedata_outputsize` (max 5000 per call); `http_timeout_seconds`; `retry_max_attempts`; optional `twelvedata_exchange` (e.g. `NASDAQ`).
- **On disk:** `{SYMBOL}_1min_{data_start}_{data_end}.parquet` — OHLCV index is bar datetime (as returned by TwelveData for `exchange_timezone`).
- **Context symbols (G3):** optional **`context_ingest.symbols`** (e.g. SPY `1min`, VIX `1day` over the same date span). Download each with **`sparkles ingest -c … -s SPY -i 1min`** (interval inferred from YAML if omitted for listed symbols). Enable only when using **`features.market_context`** (extra API credits).
- **Code:** `sparkles/data/ingest.py`, `sparkles/data/twelvedata_client.py`, `sparkles/data/retry.py`

## Historical vs live data (policy)

- **Now (Phase 1 / early iterations):** Use **batch historical** pulls only (date range in YAML, `ingest`, Parquet cache). No daemons, no scheduled “every N minutes” live polling.
- **Later:** After you are **happy with model performance** and approve a new phase, we can add **interval-based** refreshes and monitoring—still **recommendations + your logs**, not auto-trading (see **`plan.md`**).
- **PDT:** Cap is **3 day trades per 5 US business days** to stay under the usual PDT trigger—not “never day trade.”

## CLI entrypoint

After `pip install -e .` (baseline training includes **scikit-learn**; optional `[ml]` adds **xgboost**):

```bash
sparkles --help
sparkles ingest --config configs/experiments/rklb_baseline.yaml
sparkles ingest -c configs/experiments/rklb_baseline.yaml --force --verbose
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml --dry-run
sparkles report -c configs/experiments/rklb_baseline.yaml
# Optional: sparkles report -c ... --run 20260411T015314_621888Z

# After train (predictions.parquet) + journal.csv_path in YAML:
sparkles journal compare -c configs/experiments/rklb_baseline.yaml
# sparkles journal compare -c ... --run <run_id> --split val

sparkles backtest -c configs/experiments/rklb_daytrade_v2.yaml --run 20260621T161419_221266Z
sparkles backtest -c configs/experiments/rklb_daytrade_v2.yaml --run 20260621T161419_221266Z --threshold 0.35
sparkles backtest -c configs/experiments/rklb_daytrade_v2.yaml --run 20260621T161419_221266Z --sweep
sparkles meta-label train -c configs/experiments/rklb_daytrade_v2.yaml --run 20260621T161419_221266Z --primary-threshold 0.35
sparkles meta-label compare -c configs/experiments/rklb_daytrade_v2.yaml --run 20260621T161419_221266Z --primary-threshold 0.35
# sparkles backtest -c ... --no-day-trade-cap
sparkles experiments export -c configs/experiments/rklb_baseline.yaml
```

**Phase 1 smoke:** same `--config`, in order: **`ingest`** (once per range / `--force` refresh) → **`label`** → **`train`** → **`report`**. **`report`** prints whether ingest/labeled Parquet exist, **model / train / feature parameters from the current YAML**, the **latest** run’s **`metrics.json`** (accuracies, `classes`, **`features`** as stored at train time), the **last few `experiments.jsonl` rows** for the symbol (each line includes model type/solver, class_weight, feature flags, optional experiment name/notes), and optional **`--run <run_id>`** to pin a specific artifact folder.

## Config loading in code

```python
from pathlib import Path
from sparkles.config import load_experiment_config

cfg = load_experiment_config(Path("configs/experiments/rklb_baseline.yaml"))
print(cfg.symbol, cfg.min_profit_per_trade_pct)
```

## Iterations

Work proceeds by **approval-gated iterations** documented in **`plan.md`**. Do not start the next iteration without owner approval; append progress to **`plan.md`** progress log.
