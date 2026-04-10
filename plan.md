---
name: Swing ML Phase 1
overview: "Greenfield Python project: TwelveData 1m ingest with retries/cache, triple-barrier labels (15%/5% vol-scaled, tuneable min profit), train on one symbol (default RKLB), readable modules plus DEVELOPER.md; day-trade cap 3 in 5 US business days to avoid breaking PDT pattern. Historical data first—no live/scheduled API polling until owner is satisfied with model performance. Iterative roadmap with approval gates; progress in plan.md."
todos:
  - id: scaffold
    content: "Iteration 1: pyproject, package skeleton, config models, DEVELOPER.md, rklb_baseline.yaml"
    status: complete
  - id: ingest
    content: "Iteration 2: TwelveData 1m client, retry/rate limits, Parquet cache, ingest CLI"
    status: complete
  - id: vol
    content: "Iteration 3: 20-day vol series aligned to bars, tests (no lookahead)"
    status: complete
  - id: labels
    content: "Iteration 4: triple barrier + min profit + label CLI"
    status: complete
  - id: day-trade-limit
    content: "Iteration 5: day_trade_ledger (3 in 5 biz days) + tests + optional CLI hook"
    status: pending
  - id: features-train
    content: "Iteration 6: features, train.py, artifacts, experiment logging"
    status: pending
  - id: oversight
    content: "Iteration 7: polish CLI gates, README pointer, owner sign-off Phase 1"
    status: pending
isProject: false
---

# Phase 1: Single-Symbol ML System (Day-Trade Cap)

## Mandatory rules for all agents and automation

These rules apply to **every** AI agent, script, or contributor working in this repository.

1. **Approval gate:** Do **not** begin work on the **next** iteration in the roadmap below until the **project owner explicitly approves** in chat (e.g. “Approved — proceed to Iteration 3”) **or** marks that iteration as approved in this file (checkbox / date line under that iteration).
2. **Single step focus:** Complete **at most one iteration** per owner request unless the owner explicitly asks to chain multiple iterations in one go.
3. **Progress log:** After substantive work, **append** a new entry to **[Progress & change log](#progress--change-log-append-only)** at the bottom of this file. Include: date (ISO `YYYY-MM-DD`), brief summary, files/paths touched, and which iteration is now complete or blocked.
4. **Source of truth:** This **`plan.md`** is authoritative for roadmap status and recent history. If anything conflicts, follow **`plan.md`** and confirm with the owner.
5. **Cursor:** Workspace rule **`.cursor/rules/sparkles-iterative-plan.mdc`** restates the approval and logging requirements for agents in the editor.
6. **API credits:** **Preserving TwelveData API credits is crucial** (free tier). Do not add redundant fetches, high-frequency polling, or aggressive retries that burn credits. Follow **`.cursor/rules/sparkles-api-credits.mdc`** and **[METHODOLOGY.md](METHODOLOGY.md)**.

## How iterations work

- Each **Iteration N** has a **goal**, **deliverables**, and **done when** criteria.
- **Status** is one of: `not started` | `in progress` | `complete — awaiting approval for N+1` | `approved — proceed to N+1`.
- The owner advances the project by approving the next iteration in chat or by editing the **Owner approval** line under that iteration.

---

## Iteration roadmap (approval-gated)

### Iteration 0 — Planning baseline

- **Goal:** Lock architecture, constraints, and developer map in this document.
- **Status:** **complete** (baseline established; iterative gates added).
- **Owner approval to proceed to Iteration 1:** Approved (owner chat).

### Iteration 1 — Scaffold

- **Goal:** Runnable package layout, dependencies, config schema stub, `DEVELOPER.md`, `configs/experiments/rklb_baseline.yaml` (RKLB, `min_profit_per_trade_pct`), `.env.example`.
- **Deliverables:** `pyproject.toml` or `requirements.txt`, `sparkles/` package with empty modules or stubs, Pydantic config loading, documented entrypoint placeholder.
- **Done when:** `pip install -e .` (or venv + deps) succeeds; owner can find symbol and training file paths in `DEVELOPER.md`.
- **Status:** `complete — awaiting approval for Iteration 2`
- **Owner approval to proceed to Iteration 2:** `[MH ]` Date: 4-9-26

### Iteration 2 — Data ingestion

- **Goal:** TwelveData 1m **historical** fetch (backfill for `data_start`–`data_end`), retries/rate limits, Parquet cache, `ingest` CLI. **Not in scope:** daemons, scheduled “live” polling, or near-real-time loops—that waits until after the owner accepts trained-model quality (see Context: historical-first policy).
- **Deliverables:** `twelvedata_client.py`, `retry.py`, `ingest.py`, documented env var for API key.
- **Done when:** Owner can run ingest for RKLB for a configured window and see cached Parquet.
- **Status:** `complete — awaiting approval for Iteration 3`
- **Owner approval to proceed to Iteration 3:** `[MH]` Date: 4-9-26

### Iteration 3 — Volatility

- **Goal:** 20-trading-day volatility aligned to bars without lookahead.
- **Deliverables:** `features/volatility.py` (or dedicated module), unit tests for alignment.
- **Done when:** Tests pass; vol series documented in `DEVELOPER.md`.
- **Status:** `complete — awaiting approval for Iteration 4`
- **Owner approval to proceed to Iteration 4:** `[ ]` Date: ___________

### Iteration 4 — Labels

- **Goal:** Triple barrier (15% / 5% vol-scaled, `min_profit_per_trade_pct` floor), intraday path scan, `label` CLI.
- **Deliverables:** `triple_barrier.py`, `types.py`, labeled dataset output path, summary stats on CLI.
- **Done when:** Owner can run `label` and inspect class/barrier distribution.
- **Status:** `complete — awaiting approval for Iteration 5`
- **Owner approval to proceed to Iteration 5:** `[ ]` Date: ___________

### Iteration 5 — Day-trade ledger

- **Goal:** Rolling 5 US business days, max 3 day trades; tests; optional CLI dry-run.
- **Deliverables:** `sparkles/risk/day_trade_ledger.py`, tests, doc in `DEVELOPER.md`.
- **Done when:** Tests pass; ledger API documented for future backtest/live.
- **Status:** `not started`
- **Owner approval to proceed to Iteration 6:** `[ ]` Date: ___________

### Iteration 6 — Features and training

- **Goal:** Feature builders, time-based split, baseline model in **`sparkles/models/train.py`**, artifacts + run logging.
- **Deliverables:** `features/*`, `train.py`, `registry.py`, `tracking/experiments.py` (or JSONL).
- **Done when:** Owner can run `train` and get a saved model + metrics.
- **Status:** `not started`
- **Owner approval to proceed to Iteration 7:** `[ ]` Date: ___________

### Iteration 7 — Phase 1 closure

- **Goal:** CLI polish (`ingest` → `label` → `train`), optional README pointer to `DEVELOPER.md`, owner sign-off.
- **Deliverables:** End-to-end smoke path documented; frontmatter todos updated to `complete` where true.
- **Status:** `not started`
- **Owner approval (Phase 1 complete):** `[ ]` Date: ___________

---

## Context

- **Starting state:** Application code is built incrementally per the roadmap above; git repo initialized with `main` and `master` at same tip for tool compatibility.
- **Day-trade / PDT policy (design law):** The owner’s goal is to **avoid breaking the pattern day trader (PDT) band**, not to forbid day trades entirely. The program **may** use same-day round trips **only** within the cap: **at most 3 day trades in any rolling window of 5 consecutive US business days** (`max_day_trades` / `rolling_business_days` in config). That stays **below** the usual **4-in-5** trigger that applies under FINRA’s PDT framework (among other conditions). Enforcement lives in one module for **future** advisory / simulation paths. When the limit is exhausted, **do not** complete another same-day round trip (e.g. defer exit or skip—document in code and `DEVELOPER.md`).
- **Historical-first; no live polling until model sign-off:** Early iterations (**through training you are happy with**) use **batch historical** data only: pull a defined date range, cache, label, train. **Do not** add scheduled or continuous “live” TwelveData polling, streaming, or monitoring loops until the owner explicitly asks for that phase **after** they are satisfied with model performance. (Manual re-run of `ingest` for a new range is fine.)
- **No brokerage execution:** The program does **not** place orders or connect to wallets/brokers; any future “assistant” layer is **recommendations + logging** unless the owner changes scope in writing.
- **Labeling vs execution:** **Triple-barrier labels** use the **full 1-minute path from entry**, including **same-day** barrier touches. The **3-in-5 ledger** applies in **future** simulation/advisory use; optional future mode: labels that respect the ledger (defer unless requested).

## Developer guide (where to edit — readability)

Add **DEVELOPER.md** at the repo root: short “map” so you rarely hunt through the tree. It will repeat and expand on:

- **Ticker / practice symbol:** `configs/experiments/<name>.yaml` → field `symbol`. Phase 1 starter file: `configs/experiments/rklb_baseline.yaml` with **RKLB**. Use CLI `--config …`; avoid hardcoding symbols in Python.
- **Train/val dates, cache TTL, paths:** same experiment YAML.
- **Triple-barrier percents, vertical horizon, vol lookback:** same YAML; validated by Pydantic in `sparkles/config/` (e.g. `schema.py`).
- **Minimum profit per trade:** same YAML → `min_profit_per_trade_pct`. Logic only in `sparkles/labels/triple_barrier.py`; `DEVELOPER.md` states the exact formula (e.g. floor on TP after vol scaling).
- **Model family and many hparams:** YAML first; optional overrides in Python (see below).
- **Hands-on training (you edit Python):** **`sparkles/models/train.py`** — split, estimator, `fit`, save. Keep it **linear:** load → X/y → build model → fit → write artifact. Put “I’m experimenting” knobs in **`DEFAULT_TRAIN_KWARGS`** or **`build_estimator()`** at the **top** of `train.py`, with a one-line comment: “Stable hparams also in YAML under `model:`.”
- **Features:** `sparkles/features/*.py` (one theme per file).
- **Day-trade cap:** `sparkles/risk/day_trade_ledger.py` only.

**Readability conventions for `.py` files:** one short module docstring; public functions fully type-hinted; shallow nesting; no bare magic numbers (config or named constants); **`train.py`** and **`triple_barrier.py`** include a small header block: “If you change labeling horizons, see config YAML / features …”

## Default symbol for initial testing

- **Rocket Lab `RKLB`** in `configs/experiments/rklb_baseline.yaml`: `symbol: RKLB`, timezone `America/New_York`.

## Tuneable minimum profit per trade

- Config field `min_profit_per_trade_pct` (one documented convention, e.g. `0.02` = 2%).
- **Phase 1 default semantics:** floor the effective take-profit **move** after vol scaling: `effective_tp = max(min_profit_per_trade_pct, tp_move_from_vol)` (document in code + `DEVELOPER.md`). If you later add row-level filters, note that separately.
- Log this param on every training run.

## High-level architecture

```mermaid
flowchart LR
  subgraph ingest [Ingestion]
    TD[TwelveData SDK]
    Cache[Local cache Parquet]
    TD --> Cache
  end
  subgraph features [Features]
    Feat[Feature builders]
    Vol[20d vol series]
    Feat --> Vol
  end
  subgraph labels [Labels]
    TB[Triple barrier intraday path]
  end
  subgraph risk [Risk]
    DTL[Day trade ledger 3 in 5]
  end
  subgraph train [Training]
    Split[Time-based split]
    Model[Classifier in train.py]
    Track[Experiment tracking]
  end
  Cache --> Feat
  Feat --> TB
  TB --> Split
  Split --> Model
  Model --> Track
  DTL -.-> futureBacktest[Future backtest or execution]
```

## Recommended package layout (modular, PEP 8, strict typing)

All under package `sparkles/`:

- `sparkles/config/` — Pydantic models: `symbol`, dates, barrier params, `min_profit_per_trade_pct`, vol lookback, `max_day_trades: 3`, `rolling_business_days: 5`, model section, paths.
- `sparkles/data/twelvedata_client.py` — [twelvedata-python](https://github.com/twelvedata/twelvedata-python) wrapper → normalized `DataFrame`.
- `sparkles/data/ingest.py` — Chunked fetch, Parquet cache under `data/cache/`.
- `sparkles/data/retry.py` — Backoff, 429, timeouts.
- `sparkles/features/volatility.py` — 20 trading-day vol, no lookahead.
- `sparkles/labels/triple_barrier.py` — Barriers + min-profit floor; forward scan includes same session day.
- `sparkles/labels/types.py` — Outcome enums / TypedDicts.
- `sparkles/risk/day_trade_ledger.py` — Rolling 5 US business days, max 3 day-trade days; tests for weekends/holidays (calendar helper optional).
- `sparkles/models/train.py` — **Main training entrypoint you edit.**
- `sparkles/models/registry.py` — `artifacts/{symbol}/{run_id}/`.
- `sparkles/tracking/experiments.py` — MLflow or JSONL.
- `sparkles/cli.py` — `ingest`, `label`, `train`, `report`.
- `DEVELOPER.md` — Navigation guide (duplicate the bullets above in friendlier prose).

**Dependencies (indicative):** `twelvedata`, `pandas`, `numpy`, `pydantic`, `pyarrow`, `scikit-learn` and/or `xgboost`, `pyyaml`, optional `mlflow`, optional `pandas-market-calendars` for business days.

## Data ingestion (TwelveData, 1-minute)

- API key via env / gitignored `.env`.
- Chunking, cache-first, retries for timeouts and 429.

## Triple barrier (15% TP, 5% SL, 20-day vol, min profit floor)

- Vol scaling and clamps as before.
- **Effective TP move:** `max(min_profit_per_trade_pct, tp_move)`.
- Path scan: all 1m bars from entry through vertical expiry; same-day touches allowed for labels.

## Day-trade limit (3 in 5 rolling business days)

- Record each **US session date** on which a **round trip** (open and close same symbol same day) occurs.
- Before allowing a same-day close in sim or live: count such days in the rolling **5 US business days** ending at the decision date; if count ≥ **3**, **block** same-day close.
- Phase 1: ledger + unit tests + optional CLI dry-run; full simulator later.

## ML approach for Phase 1

- Classification from barrier outcomes; no feature leakage past `t0`.
- Time-ordered split; baseline in **`train.py`**.

## Oversight and workflow

1. `configs/experiments/rklb_baseline.yaml` — RKLB, barriers, `min_profit_per_trade_pct`, model hparams.
2. CLI: `ingest` → `label` → `train`.
3. Track params + metrics per run.

## Deferred

- Multi-asset, full slippage backtest, live broker APIs (still **no** auto-execution unless scope changes).
- **Live / interval ingestion** and **monitoring assistant** (journal, recommendations): after owner sign-off on model quality; tunable poll interval as a parameter when that phase is approved.

## Risk notes

- TwelveData intraday depth for RKLB; chunk if needed.
- Business-day counting: document if using a calendar library vs simplified NYSE schedule.
- RKLB is volatile; 1m barrier order can be noisy—acceptable for your test symbol.

---

## Progress & change log (append-only)

**Instructions:** Add new entries **only below** this line, newest at the bottom. Do not delete or rewrite prior entries.

| Date (ISO) | Summary | Paths / artifacts | Iteration |
|------------|---------|-------------------|-----------|
| 2026-04-07 | Iterative roadmap added: mandatory agent rules, approval gates per iteration, progress log; Cursor rule `.cursor/rules/sparkles-iterative-plan.mdc` added. Frontmatter todos remapped to iterations 1–7. | `plan.md`, `.cursor/rules/sparkles-iterative-plan.mdc` | Iteration 0 complete — **awaiting owner approval to start Iteration 1** |
| 2026-04-07 | **Iteration 1 complete:** `pyproject.toml` (deps + `sparkles` console script + ruff/mypy), full `sparkles/` package stubs, Pydantic `ExperimentConfig` + `load_experiment_config`, `configs/experiments/rklb_baseline.yaml`, `.env.example`, `DEVELOPER.md`. Verified `pip install -e ".[dev]"`, `sparkles ingest`, `ruff check`, `mypy sparkles`. | `pyproject.toml`, `sparkles/**`, `configs/experiments/rklb_baseline.yaml`, `.env.example`, `DEVELOPER.md`, `plan.md` | **Blocked until owner approves Iteration 2** (data ingestion) |
| 2026-04-07 | **Owner clarification:** PDT intent is **avoid breaking the pattern** (keep **3 day trades / 5 business days**), not zero day trades. **Historical-first:** Iteration 2+ ingest remains **batch historical** only; **no live/scheduled API polling** until owner is satisfied with trained model and approves a later phase. `DEVELOPER.md` + overview + Deferred updated. | `plan.md`, `DEVELOPER.md` | Still **blocked on Iteration 2 approval**; scope unchanged for current roadmap |
| 2026-04-09 | **Iteration 2 complete:** `retry.py` (backoff, retryable errors), `ResilientHttpClient` + `fetch_ohlcv_1min`, `ingest.run_ingest` with calendar chunking, Parquet cache + TTL, CLI `ingest --force/--verbose`. Config: `ingest_chunk_calendar_days`, `twelvedata_outputsize`, `http_timeout_seconds`, `retry_max_attempts`, `twelvedata_exchange`. Dev: `pandas-stubs`, `types-requests`, mypy overrides for `twelvedata`. Tests: `tests/test_ingest_windows.py`. | `sparkles/data/*.py`, `sparkles/cli.py`, `sparkles/config/schema.py`, `pyproject.toml`, `tests/`, `DEVELOPER.md`, `plan.md` | **Blocked until owner approves Iteration 3** (volatility) |
| 2026-04-09 | **TwelveData free-tier ingest:** pause **20s** between chunks; on per-minute credit errors sleep **~65s** then retry (not fast exponential backoff); default chunk **10** calendar days; `rklb_baseline.yaml` documents tuning. `is_per_minute_credit_exhausted_error` + `tests/test_retry_credits.py`. | `retry.py`, `twelvedata_client.py`, `ingest.py`, `schema.py`, `rklb_baseline.yaml`, `DEVELOPER.md`, `plan.md` | Iteration 3 approval unchanged |
| 2026-04-09 | **Docs + agent rules:** `.cursor/rules/sparkles-api-credits.mdc` (always apply: preserve API credits). **`METHODOLOGY.md`** (end-to-end methodology). **`README.md`** (GitHub quick start). `plan.md` mandatory rule #6; `pyproject` readme → `README.md`; `DEVELOPER.md` links updated. | `.cursor/rules/`, `METHODOLOGY.md`, `README.md`, `DEVELOPER.md`, `plan.md`, `pyproject.toml` | Iteration 3 approval unchanged |
| 2026-04-09 | **Iteration 3 complete:** `sparkles/features/volatility.py` — daily last close, rolling log-return std with `shift(1)` (no lookahead), √252 `vol_{N}d_ann` + `sigma_daily_{N}d`, `add_volatility_from_config`. Tests `tests/test_volatility.py`. `DEVELOPER.md` + `METHODOLOGY.md` updated; `features/__init__.py` exports. | `sparkles/features/`, `tests/test_volatility.py`, `DEVELOPER.md`, `METHODOLOGY.md`, `plan.md` | **Blocked until owner approves Iteration 4** (labels) |
| 2026-04-07 | **Iteration 4 complete:** `triple_barrier.py` — vol-scaled barriers (clamped), min-profit TP floor, trading-day vertical, pessimistic same-bar SL-before-TP; `BarrierOutcome` + `END_OF_DATA`. Config: `barrier_vol_scale_min` / `max`, `label_entry_stride` (default 390). CLI `sparkles label -c …` writes `{SYMBOL}_labeled_{start}_{end}_s{stride}.parquet` and prints `barrier_outcome` value counts. Tests `tests/test_triple_barrier.py`. | `sparkles/labels/`, `sparkles/config/schema.py`, `sparkles/cli.py`, `tests/test_triple_barrier.py`, `DEVELOPER.md`, `plan.md` | **Blocked until owner approves Iteration 5** (day-trade ledger) |
