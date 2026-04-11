# Phase 2 — Plan C: Position state, messaging, UI, and journal

**Parent index:** **[plan-phase2-overview.md](plan-phase2-overview.md)**.

**Depends on:** Plan B (structured **score** per bar). Plan A needed for **true** live cadence.

**Goal:** Represent **flat vs long** (manual or file-driven), switch between **entry scouting** and **exit monitoring**, combine **model scores** with **mechanistic** stop/take-profit distances from your YAML barrier params, stream **human-readable messages**, and **append** structured **journal** lines when you act (or when you configure advisory to log **signals only**).

**Still out of scope:** Broker API, auto execution.

---

## Mandatory rules

Follow **[plan-phase2-overview.md](plan-phase2-overview.md)** — Mandatory rules (Phase 2 — all plans).

---

## Iteration C1 — Position state file (manual first)

- **Goal:** Introduce a **small persisted state** (JSON or CSV) checked by the runtime: `flat` | `long`, optional `entry_price`, `entry_time`, `quantity` (optional), `stop_price` / `take_profit_price` **or** derived from YAML barrier **percent** at entry.
- **Deliverables:** Schema + reader/writer module, CLI to **set** state (`sparkles advisory position set …`) without editing JSON by hand, tests.
- **Done when:** Deleting the state file defaults safely to **flat** with a logged warning.

### How you can test (owner)

1. `advisory position set flat` then run a **score** command; output should include mode **ENTRY** or equivalent.
2. `advisory position set long --avg-price 100.0` (exact flags TBD); run again; mode **EXIT** and messages reference **distance to stop/tp** if implemented in C2.
3. Kill power mid-run; restart; state should **persist** from file.

**Owner approval to proceed to C2:** `[ ]` Date: ___________

---

## Iteration C2 — Exit helper math (mechanistic, not ML)

- **Goal:** From **current** bar and **entry** metadata, compute **distance to stop** and **distance to take-profit** using the **same** vol-scaled barrier **logic** as labels where feasible, or a **documented simpler** approximation for runtime (must be stated in docs to avoid mismatch with training labels).
- **Deliverables:** Pure functions + tests with known OHLC paths.
- **Done when:** On a toy series, hitting stop prints **STOP_TOUCH** message in replay integration test.

### How you can test (owner)

1. Use **replay** with a **fixed** long position; confirm when price crosses your stop, the stream prints an **exit alert** (even if you do not act).
2. Compare **one** scenario by hand (spreadsheet) to the program’s reported **distance %**.

**Owner approval to proceed to C3:** `[ ]` Date: ___________

---

## Iteration C3 — Policy glue (when to surface “interesting entry”)

- **Goal:** Combine **B5 ratings** with simple **gates**: e.g. do not spam when `poll_interval` fires but bar unchanged; cooldown seconds between duplicate alerts; optional **day-trade ledger** consult before encouraging a **same-day** round trip (surface **warning** only).
- **Deliverables:** YAML `advisory:` block, unit tests for cooldown and “unchanged bar” suppression.
- **Done when:** Replay produces **fewer** lines than raw per-bar scores when duplicates are filtered.

### How you can test (owner)

1. Run replay on choppy data with **cooldown 60s**; count messages vs without cooldown.
2. With ledger history flags, trigger a case where **3 day trades in 5 days** already used; confirm **WARNING** in output (**[METHODOLOGY.md](../METHODOLOGY.md)** PDT posture).

**Owner approval to proceed to C4:** `[ ]` Date: ___________

---

## Iteration C4 — Structured logging + optional desktop GUI

- **Goal:** Every emitted message is also a **JSON line** (or key=value) suitable for `grep` and future dashboards. Optional second channel: minimal **GUI** (e.g. Textual, Dear PyGui, or a local **Flask** one-page feed) subscribed to the same events.
- **Deliverables:** Log file under `logs/` or configurable path; **README-level** “how to tail”; GUI behind optional extra `[gui]` in `pyproject.toml` if dependency weight is high.
- **Done when:** Tail log during replay and see **one JSON object per event**.

### How you can test (owner)

1. `tail -f` (or `Get-Content -Wait` on Windows) the log file while replay runs; validate JSON parses in [jsonlint](https://jsonlint.com/) or `python -m json.tool` per line.
2. If GUI enabled: open the URL/window and confirm **same** events appear within **one** bar latency.

**Owner approval to proceed to C5:** `[ ]` Date: ___________

---

## Iteration C5 — Journal automation (signals and/or fills)

- **Goal:** Extend **[METHODOLOGY.md](../METHODOLOGY.md)** §9.5 journal story: **append** CSV rows for (a) **SIGNAL** rows when policy fires, (b) optional **FILL** rows when you confirm a CLI action (`advisory log fill --side buy …`). Reuse column conventions from `configs/examples/journal_trades.example.csv` where possible; new columns documented.
- **Deliverables:** Writer module, tests, `sparkles journal compare` still works or gains `--include-signals` note in docs.
- **Done when:** After a session, `journal compare` can align **signal dates** to predictions if you choose that workflow.

### How you can test (owner)

1. Run a short replay with **signal logging** on; open the CSV in Excel and verify **monotonic times** and no corrupted quoting.
2. Run `sparkles journal compare -c …` and confirm **no crash**; inspect **`journal_compare.csv`** for expected columns.
3. Optionally log a **manual** fill and confirm it appears on the **entry_date** you expect.

**Owner approval (Plan C complete):** `[ ]` Date: ___________

---

## Progress & change log (append-only)

**Instructions:** Append **only** below; newest at bottom.

| Date (ISO) | Summary | Iteration | Paths / notes |
|------------|---------|-----------|---------------|
| 2026-04-11 | Plan C drafted: C1–C5 with owner test steps. | — | `docs/plan-phase2-03-state-ui-journal.md` |
