# Sparkles — methodology

This document describes **what** we are building and **why**, at a level suitable for you or a future contributor. Operational “where to click” detail lives in **[README.md](README.md)** and **[DEVELOPER.md](DEVELOPER.md)**. The **approval-gated roadmap** and history live in **[plan.md](plan.md)**.

---

## Purpose and general use (read this first)

**What the program is for — Phase 1 (now):** Sparkles is an **offline research pipeline** for **one US symbol at a time**. You pull a chosen **historical** 1m range from **TwelveData**, cache it as **Parquet**, build **triple-barrier labels** (mechanistic outcomes after a hypothetical long: take-profit, stop-loss, time exit, or end-of-data), and **train models** on those labels. There is **no** long-running process that watches the market, **no** broker connection, and **no** auto-trading.

**What the models are for:** Trained models are meant to **inform decisions** in a **future** advisory layer (e.g. whether a setup resembles past labeled paths). They do **not** by themselves mean “good buy” or “good sell”; labels are **rule-based targets** for learning, not trade recommendations.

**What comes later (only after you approve it):** A separate phase may add **interval or live data refresh**, a **journal** (e.g. you log when you actually enter or exit), and an assistant that **recommends** entries/exits and respects **stops** and **day-trade limits** — still **you** confirm actions unless scope changes in writing. Phase 1 deliberately stays **batch historical** until you are satisfied with model quality (see §2.1).

**Risk posture:** A **day-trade ledger** encodes **at most 3 day trades per 5 US business days** (configurable) so you can stay **under** the usual **PDT** pattern band; use it from backtests and, later, from any live-style helper you build.

---

## 1. Purpose (detail)

Sparkles is a **research and assistant-oriented** pipeline for **one US equity at a time** (default experiment: **RKLB**). It is designed to:

- Pull **historical 1-minute OHLCV** from **TwelveData**, cache it locally, label and train models on that data.
- Enforce a **conservative day-trade budget** (**at most 3 day trades per 5 US business days**) so the owner can stay **below** the usual **pattern day trader (PDT)** trigger band (FINRA framing often uses **4** day trades in 5 business days, among other conditions).
- **Not** connect to brokers, wallets, or order APIs. **No automatic trading.** Any future “assistant” layer is **recommendations plus your manual logs**, unless scope is explicitly changed in writing.

---

## 2. Data philosophy

### 2.1 Historical first

Until the owner is **satisfied with model performance**, work stays **batch historical**: define `data_start` / `data_end`, run **`sparkles ingest`**, get a **Parquet** cache file. **No** daemons, **no** scheduled live polling, **no** streaming TwelveData loops—those belong to a **later approved phase** (see `plan.md`).

### 2.2 Storage format (Parquet)

Ingested bars are stored as **Apache Parquet** under `data/cache/` (configurable via `paths.cache_dir`). Parquet is a **columnar binary table** format: think **one pandas `DataFrame`** (or one SQL table) per file—efficient and standard for ML pipelines. It is **not** JSON on disk; load with `pandas.read_parquet` in Python.

### 2.3 Cache behavior

- If the cache file exists and is **newer than `cache_ttl_hours`**, ingest **skips** the API and reuses the file.
- **`--force`** (or an expired TTL) **re-downloads** the full configured range and **overwrites** that Parquet file (no silent append).

### 2.4 API credits (critical)

TwelveData **free tier** enforces tight **per-minute** (and daily) **API credit** limits. The codebase **throttles** between chunk requests, uses **fewer/larger calendar chunks** where safe, and on **per-minute credit exhaustion** waits **~65 seconds** before retrying instead of hammering short backoffs.

**Agents and contributors must preserve credits** (see **`.cursor/rules/sparkles-api-credits.mdc`**): avoid redundant calls, respect cache TTL, and do not add high-frequency API features without explicit design approval.

### 2.5 Secrets

`TWELVEDATA_API_KEY` is read from the **environment** (or a **gitignored** `.env`). Keys must **never** be committed. Market cache and Parquet files are **gitignored** by default so large data and keys do not leak to GitHub.

---

## 3. Configuration-driven experiments

Each run is driven by a **YAML experiment file** (e.g. `configs/experiments/rklb_baseline.yaml`) validated by **Pydantic** (`sparkles/config/schema.py`). It holds:

- Symbol, exchange timezone, data date range, train/val windows (when used).
- Triple-barrier **base** take-profit and stop (e.g. 15% / 5%), **volatility lookback**, **vertical** (time) barrier, and **`min_profit_per_trade_pct`** (floor on the effective take-profit move after vol scaling).
- Day-trade cap parameters (`max_day_trades`, `rolling_business_days`).
- Ingest throttling and TwelveData options (chunk days, sleep between chunks, per-minute credit wait, `outputsize`, timeouts, retries).
- Paths for cache and future artifacts.

---

## 4. Pipeline stages (Phase 1 roadmap)

Work proceeds in **iterations** documented in **`plan.md`**; the next stage starts only after **owner approval**.

| Stage | Role |
|--------|------|
| **Ingest** | TwelveData 1m historical → normalized OHLCV → Parquet cache. |
| **Volatility** | Daily-close log returns → rolling std over `vol_lookback_trading_days`, **`shift(1)`**, √252 annualization; broadcast to every 1m bar on that session date (`sparkles/features/volatility.py`). |
| **Labels (triple barrier)** | For each candidate entry time: upper barrier (take-profit path), lower barrier (stop), vertical barrier (max holding time). Moves scaled by **recent volatility** vs a reference; effective TP floored by **`min_profit_per_trade_pct`**. Path uses **full 1m forward path**, including **same-day** touches (labels match intraday reality). |
| **Day-trade ledger** | Rolling **weekday** window (`rolling_business_days`), **≤ `max_day_trades`** day-trade **events**; for backtests and **future** simulation/advisory logic (holidays not excluded in v1). |
| **Features + train** | Entry-only feature join (`sparkles/features/dataset.py`); session-date train/val from YAML; **`sparkles train`** → `artifacts/{SYMBOL}/{run_id}/model_bundle.joblib` + `metrics.json` + **`experiments.jsonl`**. |
| **Closure** | **`sparkles report`**, train prints headline metrics, README/DEVELOPER smoke path; formal Phase 1 sign-off in **`plan.md`**. |

---

## 5. Machine-learning framing

- **Supervised learning** on **triple-barrier outcomes** (e.g. which barrier hit first, or derived binary/ternary targets).
- **Validation**: time-based / walk-forward splits—not random row splits—for series data.
- **Model code** is intentionally **editable** in one place (`train.py`) with hyperparameters mirrored in YAML where stable.

---

## 6. Product vision (later phases)

This overlaps with **[Purpose and general use](#purpose-and-general-use-read-this-first)** above: after the owner trusts the model, a separate phase may add **interval-based** refresh of data and a **manual** buy/sell journal with **recommendations only**—still **no** auto-execution unless scope changes in writing.

---

## 7. Engineering standards

- **Python 3.10+**, **PEP 8**, **strict typing** on new code, modular packages under `sparkles/`.
- **CLI**: `sparkles` (Typer) with `ingest`, `label`, `risk`, `train`, `report`.
- **Quality**: `ruff`, `mypy`, `pytest` in optional `[dev]` install.

---

## 8. Document map

| File | Use |
|------|-----|
| **README.md** | Repo overview and quick start (GitHub landing). |
| **METHODOLOGY.md** | This file — concepts and methodology. |
| **DEVELOPER.md** | Where to edit symbol, training file, ingest knobs. |
| **plan.md** | Iterations, approvals, append-only progress log. |
| **[docs/README.md](docs/README.md)** | Index of files under `docs/`. |
| **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)** | Post–Phase 1 roadmap: models, features, YAML expansion (append-only log). |

---

## 9. Terminal examples (copy-paste)

From the **repository root**, with your experiment YAML (default path used below). Set **`TWELVEDATA_API_KEY`** first (environment or `.env` per **`.env.example`**).

**Windows PowerShell (API key for the current session):**

```powershell
$env:TWELVEDATA_API_KEY = "your_key_here"
cd "C:\path\to\Sparkles"
```

**Install (once per venv):**

```bash
python -m pip install -e ".[dev]"
```

**Historical ingest** (writes 1m Parquet under `data/cache/`; prints absolute path):

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml
```

Verbose chunk logging; **force** full re-download ignoring cache age:

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
sparkles ingest -c configs/experiments/rklb_baseline.yaml --force -v
```

**Triple-barrier labels** (needs ingest Parquet for the same `symbol` + `data_start` / `data_end`; prints path + outcome counts):

```bash
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles label -c configs/experiments/rklb_baseline.yaml -v
```

**Day-trade cap dry-run** (uses `max_day_trades` / `rolling_business_days` from YAML):

```bash
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml --as-of 2026-04-01
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml --as-of 2026-04-01 --history 2026-03-25,2026-03-26
```

**Train** (needs labeled Parquet + `train_*` / `val_*` in YAML; prints run dir + headline metrics):

```bash
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml -v
```

**Report** (ingest/label paths, latest `metrics.json`, tail of `experiments.jsonl`; optional specific run folder):

```bash
sparkles report -c configs/experiments/rklb_baseline.yaml
sparkles report -c configs/experiments/rklb_baseline.yaml --run 20260411T015314_621888Z
```

**Typical Phase 1 order** (same `--config` throughout):

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles report -c configs/experiments/rklb_baseline.yaml
```

**Help:**

```bash
sparkles --help
sparkles ingest --help
sparkles label --help
sparkles risk day-trades --help
sparkles train --help
sparkles report --help
```
