# Sparkles — methodology

This document describes **what** we are building and **why**, at a level suitable for you or a future contributor. Operational “where to click” detail lives in **[README.md](README.md)** and **[DEVELOPER.md](DEVELOPER.md)**. The **approval-gated roadmap** and history live in **[plan.md](plan.md)**.

---

## 1. Purpose

Sparkles is a **research and assistant-oriented** pipeline for **one US equity at a time** (default experiment: **RKLB**). It is designed to:

- Pull **historical 1-minute OHLCV** from **TwelveData**, cache it locally, and later label and train models on that data.
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
| **Day-trade ledger** | Rolling **5 US business days**, **≤ 3** day-trade days; used for **future** simulation/advisory logic, not for rewriting labels in Phase 1 unless requested later. |
| **Features + train** | Leakage-safe features, time-ordered split, baseline classifier in **`sparkles/models/train.py`**, saved artifacts + run logging. |
| **Closure** | CLI polish, smoke path, owner sign-off. |

---

## 5. Machine-learning framing

- **Supervised learning** on **triple-barrier outcomes** (e.g. which barrier hit first, or derived binary/ternary targets).
- **Validation**: time-based / walk-forward splits—not random row splits—for series data.
- **Model code** is intentionally **editable** in one place (`train.py`) with hyperparameters mirrored in YAML where stable.

---

## 6. Product vision (later phases)

After the owner trusts the model, a separate phase may add **interval-based** refresh of data and a **manual** buy/sell journal with **recommendations only**—still **no** auto-execution unless scope changes in writing.

---

## 7. Engineering standards

- **Python 3.10+**, **PEP 8**, **strict typing** on new code, modular packages under `sparkles/`.
- **CLI**: `sparkles` (Typer) with subcommands `ingest`, `label`, `train`, `report` (stubs advance per roadmap).
- **Quality**: `ruff`, `mypy`, `pytest` in optional `[dev]` install.

---

## 8. Document map

| File | Use |
|------|-----|
| **README.md** | Repo overview and quick start (GitHub landing). |
| **METHODOLOGY.md** | This file — concepts and methodology. |
| **DEVELOPER.md** | Where to edit symbol, training file, ingest knobs. |
| **plan.md** | Iterations, approvals, append-only progress log. |
