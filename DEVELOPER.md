# Sparkles developer map

Quick reference for **where to change what** in Phase 1. Conceptual methodology: **[METHODOLOGY.md](METHODOLOGY.md)**. Roadmap and approvals: **`plan.md`**. GitHub overview: **[README.md](README.md)**. **ML expansion** (models, features, YAML roadmap): **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)**.

## Repository layout

| Path | Role |
|------|------|
| **`sparkles/`** | Package: `config`, `data`, `features`, `journal`, `labels`, `models`, `reporting`, `risk`, `tracking` (`experiments.py`, `experiments_csv.py`), `cli.py`. |
| **`configs/experiments/`** | Experiment YAML (e.g. `rklb_baseline.yaml`); validated by `sparkles/config/schema.py`. |
| **`tests/`** | `pytest` only; mirror modules under `sparkles/` where practical. |
| **`scripts/`** | Non-installed helpers (e.g. `quick_try_vol.py`); run with `python scripts/...`. |
| **`docs/`** | Longer-lived docs beyond root summaries (`ML_EXPANSION.md`, this layout index in `docs/README.md`). |
| **`data/cache/`** | Parquet from **`ingest`** / **`label`** (gitignored). |
| **`data/journal/`** | Optional personal trade CSVs (see README there; `*.csv` gitignored). |
| **`artifacts/`** | Training runs + `experiments.jsonl` from **`train`** (gitignored). |
| **Root** | `README.md`, `METHODOLOGY.md`, `DEVELOPER.md`, `plan.md`, `pyproject.toml`, `.env.example`, **`Sparkles.code-workspace`** (optional VS Code multi-root). |
| **`.cursor/rules/`** | Cursor agent rules (API credits, iterative plan). |

## Practice symbol (ticker)

- **File:** `configs/experiments/rklb_baseline.yaml` (or your own copy under `configs/experiments/`)
- **Field:** `symbol` (default **RKLB**)
- **CLI:** pass `--config path/to/experiment.yaml` to `sparkles` subcommands

## Training code (Python you edit often)

- **File:** `sparkles/models/train.py` — **`run_train(cfg)`** loads labeled + ingest Parquets, **`build_feature_matrix`**, time-splits, fits **`model.type`** via **`sparkles/models/estimators.py`**, writes **`model_bundle.joblib`**, **`metrics.json`**, **`experiment_config.json`** (full YAML-equivalent snapshot), and (by default) **`predictions.parquet`** per-row **`train.export_predictions`**: **`val`** (validation rows only), **`all`** (train+val), or **`none`**. Columns include **`entry_time`**, **`session_date`**, **`split`**, **`y_true`**, **`y_pred`**, per-class **`proba_*`**, **`max_proba`** when the estimator supports **`predict_proba`**. Path: **`{artifacts_dir}/{SYMBOL}/{run_id}/`**. Appends **`experiments.jsonl`** (includes nested **`experiment_config`** plus headline fields). CSV export: **`sparkles experiments export -c …`** → **`artifacts/training_log.csv`** by default; **`--all-symbols`** exports every symbol in the log.
- **Estimators:** `sparkles/models/estimators.py` — **`build_estimator`**: `logistic_regression` (sklearn, core deps) or **`xgboost_classifier`** (install **`pip install -e ".[ml]"`**). XGBoost hyperparameters: **`model.xgb_n_estimators`**, **`xgb_max_depth`**, **`xgb_learning_rate`**, **`xgb_subsample`**, **`xgb_colsample_bytree`**, **`xgb_min_child_weight`**. YAML **`model.class_weight`** maps to sklearn for logistic regression and to **`sample_weight`** for XGBoost when not null.
- **Registry:** `sparkles/models/registry.py` — `new_run_id`, `run_artifact_dir`, `save_bundle`, `save_json`.
- **Features at entry only:** Controlled by **`features:`** in YAML (`FeatureConfig` in `schema.py`). Each flag includes a builder group from **`sparkles/features/registry.py`** (see **`sparkles/features/builders.py`**):
  - **`log_entry_close`:** column `log_entry_close` — `log(entry_close)` from the label row.
  - **`label_geometry`:** `sigma_ann_at_entry`, `vol_scale_ratio`, `tp_move_effective`, `sl_move` — barrier/vol snapshot from labeling.
  - **`intraday_range_pct`:** `(high − low) / entry_close` on the **entry bar** from ingest OHLCV (requires `high`, `low`, `close` on OHLCV).
  - **`log1p_volume`:** `log1p(volume)` on the entry bar (volume optional on OHLCV; missing → zeros).
  All default **true** (Phase 1 column set). No future path / `bars_forward` in the feature matrix.
- **Config:** `model.*`, `train.*` (includes **`export_predictions`**), and **`features.*`**. Optional **`journal:`** — **`csv_path`**: your trade log (repo-relative or absolute). **CSV:** date column **`entry_date`**, **`date`**, **`open_date`**, or **`entry`** (ISO). Optional **`symbol`** / **`ticker`** filters rows to the experiment **`symbol`**. Extra columns (`exit_date`, `shares`, `pnl_pct`, `notes`, …) are kept in the merge output. **Long holds:** one row per **open** is fine; compare aligns on **entry session date** to aggregated model predictions that day (not daily P&L over the hold).
- **Journal compare:** `sparkles journal compare -c …` — loads **`journal.csv_path`**, latest run with **`predictions.parquet`** (or **`--run <run_id>`**), aggregates predictions by **`session_date`** for **`--split val`** (default), **`train`**, or **`both`**, left-joins journal **`entry_date`**, writes **`journal_compare.csv`** in the run folder. Template: **`configs/examples/journal_trades.example.csv`**.
- **Prerequisites:** `sparkles ingest` then `sparkles label` for the same `symbol`, `data_start`, `data_end`, and `label_entry_stride` as in the YAML.

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

- **CLI:** `sparkles ingest -c configs/experiments/rklb_baseline.yaml` prints the absolute path to the Parquet file written under `cache_dir`.
- **Flags:** `--force` / `-f` bypasses `cache_ttl_hours` and re-downloads the full `data_start`–`data_end` range; `-v` enables verbose chunk logging.
- **Config:** `ingest_chunk_calendar_days` (default **10** — fewer HTTP calls); `ingest_sleep_seconds_between_chunks` (default **20** s — stay under free-tier **~8 API credits/minute**); `twelvedata_per_minute_credit_wait_seconds` (default **65** — wait for the next minute when TwelveData returns a per-minute credit error, instead of fast retries); `twelvedata_outputsize` (max 5000 per call); `http_timeout_seconds`; `retry_max_attempts`; optional `twelvedata_exchange` (e.g. `NASDAQ`).
- **On disk:** `{SYMBOL}_1min_{data_start}_{data_end}.parquet` — OHLCV index is bar datetime (as returned by TwelveData for `exchange_timezone`).
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
sparkles report -c configs/experiments/rklb_baseline.yaml
# Optional: sparkles report -c ... --run 20260411T015314_621888Z

# After train (predictions.parquet) + journal.csv_path in YAML:
sparkles journal compare -c configs/experiments/rklb_baseline.yaml
# sparkles journal compare -c ... --run <run_id> --split val
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
