# Phase 2 — Plan A: Live-ish data path and API credits

**Parent index:** **[plan-phase2-overview.md](plan-phase2-overview.md)**.

**Goal:** Move from “only full-range batch ingest” to a **controlled** way to bring **recent** 1-minute bars into the workspace during **allowed session windows** (regular and/or extended hours, depending on TwelveData symbol coverage and your subscription), while **preserving API credits** and keeping behavior **testable** and **reproducible**.

**Out of scope for Plan A:** Model scoring, position files, and GUI (Plans B and C).

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
