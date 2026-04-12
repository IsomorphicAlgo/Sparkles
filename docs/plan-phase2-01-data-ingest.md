# Phase 2 — Plan A: Live-ish data path and API credits

**Parent index:** **[plan-phase2-overview.md](plan-phase2-overview.md)**.

**Goal:** Move from “only full-range batch ingest” to a **controlled** way to bring **recent** 1-minute bars into the workspace during **allowed session windows** (regular and/or extended hours, depending on TwelveData symbol coverage and your subscription), while **preserving API credits** and keeping behavior **testable** and **reproducible**.

**Out of scope for Plan A:** Model scoring, position files, and GUI (Plans B and C).

---

## Design specification (Iteration A1 — locked)

This section records **what “live” means** in Sparkles before implementation (A3+). It is the **paper design** for Plan A.

### Mode: interval HTTP refresh (not websocket in v1)

- **v1:** A **polling** loop (A4) calls an **incremental fetch** (A3) on a **timer**: `poll_interval_seconds` (minimum **60s** in schema; default **120s**) between **completed** refresh attempts, not between every sub-chunk inside one refresh (chunk sleeps reuse existing ingest throttles).
- **Future:** WebSockets or push feeds are **out of scope** until explicitly approved; the same YAML block can later gain a `transport: http | …` field if needed.

### Data layout: separate recent file vs main cache

- **Default (`merge_strategy: separate_recent_parquet`):** Keep the existing **batch** Parquet `{SYMBOL}_1min_{data_start}_{data_end}.parquet` as the **historical** truth from `sparkles ingest`. Writes **recent** bars to a **sidecar** file (naming finalized in A3), e.g. `{SYMBOL}_1min_recent.parquet`, **deduped by bar timestamp**, so refreshes never silently **overwrite** years of backfill.
- **Optional (`merge_into_main_cache`):** Append new bars into the main cache file (more foot-gun risk; only for advanced users who accept longer Parquet rewrites).

### Lookback per refresh

- **`refresh_lookback_calendar_days`** (default **2**, max **31**): bounds each refresh request to **recent calendar days** of 1m data, limiting payload size and API credits per tick.

### Session window and extended hours

- **`session_start_local` / `session_end_local`:** Optional **HH:MM** wall-clock in **`exchange_timezone`**. **Both set or both omitted.** If omitted, the loop does **not** gate on clock (implementation still obeys API in A4). **Overnight windows** (e.g. 22:00–04:00) are **not** supported in v1—use full-day (omit both) or a same-day span (e.g. **04:00–20:00** for a long US equity window).
- **`include_extended_hours`:** When **true**, implementation (A3) will request extended-hours 1m where TwelveData supports it for the symbol; when **false**, rely on the provider default (often **regular hours only**). Exact API parameters are fixed in A3.

### Single symbol

- Phase 2 refresh uses the **same** experiment YAML as Phase 1: **one `symbol` per config file**. Multi-ticker orchestration is out of scope.

### Relationship to `cache_ttl_hours` (batch ingest)

- **`cache_ttl_hours`** applies only to **historical** `sparkles ingest` skipping re-download of the **full** `data_start`–`data_end` range.
- **Live refresh** does **not** use TTL to block new bars: the hot path is the **sidecar** / incremental merge. Re-running full `sparkles ingest` still respects TTL for the big file unless **`--force`**.

### Failure modes (behavior in A3+)

| Situation | Intended behavior |
|-----------|-------------------|
| Empty or error response from API | Log warning / error; **do not** truncate existing Parquet; optional retry via `retry.py`. |
| Partial day / gaps | Merge what returned; **dedupe** by timestamp. |
| Clock / timezone | Session window evaluated in **`exchange_timezone`** (same as labels). |
| Stale data | Document **`max_staleness_minutes`** in a later iteration if needed; not required for A1. |

### Credit posture (defaults)

- Default **`poll_interval_seconds` = 120** and **minimum 60** keep headroom above typical **free-tier** per-minute limits (see **[METHODOLOGY.md](../METHODOLOGY.md)** §2.4). **Do not** set very low intervals without a paid tier.

### Config surface (Iteration A1)

- Implemented as validated **`live_ingest:`** in experiment YAML (`LiveIngestConfig` in **`sparkles/config/schema.py`**). **`enabled` defaults to `false`** so Phase 1 workflows are unchanged until you opt in.

---

## Mandatory rules

Follow **[plan-phase2-overview.md](plan-phase2-overview.md)** — Mandatory rules (Phase 2 — all plans).

---

## Iteration A1 — Requirements and “paper design”

- **Goal:** Lock **what** “live” means for Sparkles (interval pull vs append-only file vs future websocket), **when** pulls may run (timezone + optional clock window in YAML), and **how** that coexists with existing Parquet cache files.
- **Deliverables:** Short **design section** (can live at the top of this file or in a new `docs/` note linked here): default **minimum seconds between API calls**, maximum **lookback** per refresh (e.g. last 1–5 trading days of 1m bars), and whether refresh **merges into** the existing cache file or writes a **separate** `*_recent.parquet` that downstream code reads together with historical cache. No production polling loop yet—**design + config keys** agreed in YAML schema (may be stubbed).
- **Done when:** Design answers: (1) single symbol only Phase 2 yes/no, (2) extended hours inclusion, (3) overlap with `cache_ttl_hours` behavior, (4) failure modes (empty response, stale clock).

### How you can test (owner)

1. Read the design section and confirm it matches your **intended trading window** (e.g. premarket start → your end time, exchange timezone).
2. Check that **default** proposed poll interval is **not** below TwelveData’s safe floor for your tier (compare **[METHODOLOGY.md](../METHODOLOGY.md)** §2.4).
3. **Sign-off:** Note approval on the line below.

**Owner approval to proceed to A2:** `[ ]` Date: ___________

---

## Iteration A2 — Config surface (YAML + validation)

- **Goal:** Add **validated** experiment fields for refresh mode (e.g. `live_ingest.enabled`, `live_ingest.poll_interval_seconds`, `live_ingest.session_*` or `allowed_windows`, `live_ingest.merge_strategy`). Fields may be **no-op** until A3 implements behavior.
- **Deliverables:** Pydantic updates in `sparkles/config/schema.py`, commented examples in `configs/experiments/` (or `configs/examples/`), unit tests that invalid YAML fails fast.
- **Done when:** `load_experiment_config` accepts the new block; `sparkles report` (or a tiny `sparkles config dump` if you add it) shows parsed values.

### How you can test (owner)

1. Copy your baseline YAML, add the new `live_ingest:` (or agreed name) block from the example.
2. Run: `sparkles report -c path/to/your.yaml` and confirm new fields appear in the printed summary (or run the config-load test command documented in the implementation PR).
3. Deliberately set an **invalid** value (e.g. negative interval) and confirm the CLI exits with a **clear validation error**.

**Owner approval to proceed to A3:** `[ ]` Date: ___________

---

## Iteration A3 — Incremental fetch implementation

- **Goal:** Implement **one** supported strategy from A1 (recommend starting with **interval HTTP fetch** of a **short** recent window, reusing `twelvedata` client patterns from batch ingest). **Deduplicate** by bar timestamp; **append or merge** per A1 design.
- **Deliverables:** Module under `sparkles/data/` (e.g. `refresh.py` or extend `ingest.py`), functions covered by **mocked HTTP** tests (no real API in CI).
- **Done when:** From a known small fixture, merged Parquet has **no duplicate indices** and **monotonic** time index.

### How you can test (owner)

1. **Automated:** `pytest` for the new tests (maintainer runs in CI); you run `python -m pytest tests/...` for the new file locally.
2. **Manual (costs API credits):** With `TWELVEDATA_API_KEY` set, run the new CLI (e.g. `sparkles ingest-refresh` or documented flag on `ingest`) **once** with a **short** window and `-v`; confirm log shows **one** bounded fetch, not a loop.
3. Open the output Parquet in Python or run `sparkles report` and confirm **row count increased** or recent timestamps moved forward without destroying older history.

**Owner approval to proceed to A4:** `[ ]` Date: ___________

---

## Iteration A4 — Optional daemon / scheduled loop (dev-safe)

- **Goal:** A **long-running** command that wakes on `poll_interval_seconds`, **only** calls refresh when inside the configured **session window**, and **sleeps** otherwise. Must log **each** wake and **each** skip reason (outside window, cache still fresh, etc.).
- **Deliverables:** CLI entry (e.g. `sparkles run refresh-loop` or `sparkles live refresh`), graceful **Ctrl+C** shutdown, structured log lines.
- **Done when:** Process runs for **15+ minutes** in a test without exceeding documented max calls/hour for your tier **in dry configuration** (use a long interval for the smoke test).

### How you can test (owner)

1. Set `poll_interval_seconds` to something **large** (e.g. 300) and a **narrow** window so most ticks are “skip”; run the loop for 10 minutes and confirm logs show **skips**, not repeated downloads.
2. Temporarily set interval **short** only on a **paper** key or mock (if provided); otherwise rely on unit tests—**do not** burn credits with 5-second polling.
3. Stop with **Ctrl+C**; confirm clean exit (no stack trace) and no corrupt Parquet.

**Owner approval to proceed to A5:** `[ ]` Date: ___________

---

## Iteration A5 — Hardening and documentation

- **Goal:** Document **credit budget** math (rough calls per hour), **failure** retries (reuse `retry.py` behavior), and **operational** checklist. Wire **DEVELOPER.md** / **METHODOLOGY.md** pointers if behavior is user-facing.
- **Deliverables:** Doc sections + example YAML for “extended morning session” profile; optional `.env` flags if needed.
- **Done when:** A new reader can run Plan A safely without guessing intervals.

### How you can test (owner)

1. Follow the **operational checklist** from a cold start (new terminal, key in env) and complete one **full** session dry-run on paper settings.
2. Run `sparkles report -c …` before and after a refresh and confirm **paths and recency** match expectations.

**Owner approval (Plan A complete):** `[ ]` Date: ___________

---

## Progress & change log (append-only)

**Instructions:** Append **only** below; newest at bottom.

| Date (ISO) | Summary | Iteration | Paths / notes |
|------------|---------|-----------|---------------|
| 2026-04-11 | Plan A drafted: A1–A5 with owner test steps. | — | `docs/plan-phase2-01-data-ingest.md` |
| 2026-04-12 | **A1:** Design section (this file), **`LiveIngestConfig`** + **`live_ingest`** on **`ExperimentConfig`**, report line, commented YAML example, tests. | A1 | `sparkles/config/schema.py`, `sparkles/reporting/summary.py`, `configs/experiments/rklb_baseline.yaml`, `tests/test_live_ingest_config.py` |
