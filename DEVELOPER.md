# Sparkles developer map

Quick reference for **where to change what** in Phase 1. Conceptual methodology: **[METHODOLOGY.md](METHODOLOGY.md)**. Roadmap and approvals: **`plan.md`**. GitHub overview: **[README.md](README.md)**.

## Practice symbol (ticker)

- **File:** `configs/experiments/rklb_baseline.yaml` (or your own copy under `configs/experiments/`)
- **Field:** `symbol` (default **RKLB**)
- **CLI:** pass `--config path/to/experiment.yaml` to `sparkles` subcommands

## Training code (Python you edit often)

- **File:** `sparkles/models/train.py`
- Use **`DEFAULT_TRAIN_KWARGS`** or **`build_estimator()`** at the top of that file for quick experiments; mirror stable settings under `model:` in the same YAML.

## Labeling and minimum profit per trade

- **Config:** `min_profit_per_trade_pct` in experiment YAML (fraction, e.g. `0.02` = **2%**)
- **Implementation (Iteration 4):** `sparkles/labels/triple_barrier.py` — planned semantics: floor the vol-scaled take-profit move:
  `effective_tp = max(min_profit_per_trade_pct, tp_move_from_vol)`

## Barriers, vol lookback, vertical horizon

- **Config:** `profit_barrier_base`, `stop_loss_base`, `vol_lookback_trading_days`, `vertical_max_trading_days` in experiment YAML
- **Vol series (Iteration 3):** `sparkles/features/volatility.py`

## Day-trade cap (3 in 5 US business days)

- **Implementation (Iteration 5):** `sparkles/risk/day_trade_ledger.py` only
- **Config:** `max_day_trades`, `rolling_business_days`

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

After `pip install -e .` (and optional `[ml]` extra for sklearn/xgboost later):

```bash
sparkles --help
sparkles ingest --config configs/experiments/rklb_baseline.yaml
sparkles ingest -c configs/experiments/rklb_baseline.yaml --force --verbose
```

`label` and `train` remain stubs until later iterations.

## Config loading in code

```python
from pathlib import Path
from sparkles.config import load_experiment_config

cfg = load_experiment_config(Path("configs/experiments/rklb_baseline.yaml"))
print(cfg.symbol, cfg.min_profit_per_trade_pct)
```

## Iterations

Work proceeds by **approval-gated iterations** documented in **`plan.md`**. Do not start the next iteration without owner approval; append progress to **`plan.md`** progress log.
