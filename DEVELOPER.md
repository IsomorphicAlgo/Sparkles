# Sparkles developer map

Quick reference for **where to change what** in Phase 1. Conceptual methodology: **[METHODOLOGY.md](METHODOLOGY.md)**. Roadmap and approvals: **`plan.md`**. GitHub overview: **[README.md](README.md)**.

## Practice symbol (ticker)

- **File:** `configs/experiments/rklb_baseline.yaml` (or your own copy under `configs/experiments/`)
- **Field:** `symbol` (default **RKLB**)
- **CLI:** pass `--config path/to/experiment.yaml` to `sparkles` subcommands

## Training code (Python you edit often)

- **File:** `sparkles/models/train.py` — **`run_train(cfg)`** loads labeled + ingest Parquets, **`build_feature_matrix`** (`sparkles/features/dataset.py`) joins on `entry_time`, splits by **US session date** using **`train_start` / `train_end` / `val_start` / `val_end`** (all four required for training), fits **`model.type`** `logistic_regression` (sklearn), writes **`model_bundle.joblib`** + **`metrics.json`** under **`{artifacts_dir}/{SYMBOL}/{run_id}/`**, and appends a line to **`{artifacts_dir}/experiments.jsonl`**.
- **Registry:** `sparkles/models/registry.py` — `new_run_id`, `run_artifact_dir`, `save_bundle`, `save_json`.
- **Features at entry only:** `log_entry_close`, `sigma_ann_at_entry`, `vol_scale_ratio`, `tp_move_effective`, `sl_move`, `intraday_range_pct`, `log1p_volume` (no future path / `bars_forward` in X).
- **Config:** `model.random_seed`, `model.logistic_c`, `model.max_iter` (see `ModelConfig` in `schema.py`). Use **`DEFAULT_TRAIN_KWARGS`** in `train.py` for scratch overrides not yet in YAML.
- **Prerequisites:** `sparkles ingest` then `sparkles label` for the same `symbol`, `data_start`, `data_end`, and `label_entry_stride` as in the YAML.

## Labeling and minimum profit per trade

- **Config:** `min_profit_per_trade_pct` in experiment YAML (fraction, e.g. `0.02` = **2%**)
- **Implementation:** `sparkles/labels/triple_barrier.py` — vol-scaled TP/SL (ratio `sigma_t / sigma_ref` clamped by `barrier_vol_scale_min` / `barrier_vol_scale_max`), then **effective TP move** `max(min_profit_per_trade_pct, tp_move_from_vol)`. Forward scan on 1m `high`/`low`; on a tie in the same bar, **stop** is checked before **take-profit** (pessimistic long). Vertical exit uses **trading-day** count from entry (`vertical_max_trading_days`). Entries are every `label_entry_stride` bars (default **390** ≈ one per regular session).
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
```

**Phase 1 smoke:** same `--config`, in order: **`ingest`** (once per range / `--force` refresh) → **`label`** → **`train`** → **`report`**. **`report`** prints whether ingest/labeled Parquet exist, the **latest** training run under **`artifacts/{SYMBOL}/`** (or **`--run <run_id>`**), and the last few **`experiments.jsonl`** rows for that symbol.

## Config loading in code

```python
from pathlib import Path
from sparkles.config import load_experiment_config

cfg = load_experiment_config(Path("configs/experiments/rklb_baseline.yaml"))
print(cfg.symbol, cfg.min_profit_per_trade_pct)
```

## Iterations

Work proceeds by **approval-gated iterations** documented in **`plan.md`**. Do not start the next iteration without owner approval; append progress to **`plan.md`** progress log.
