---
name: Phase 2 — Live advisory (overview)
overview: "Master index and shared rules for turning Sparkles from batch offline research into an optional interval/live assistant: refreshed data, per-bar inference, position-aware entry/exit guidance, logging and UI—without broker execution unless scope changes in writing."
isProject: false
---

# Phase 2 overview: toward a live (or interval) assistant

**Phase 1** (authoritative history and closure): **[plan.md](../plan.md)**.

This **Phase 2** set of documents describes **iterative work** to support your vision: start during a chosen session window (for example extended hours), refresh or stream data within **API credit** limits, score each new bar for **entry vs risk** when flat, switch to **exit-oriented** logic when a position is recorded, surface messages in **real time**, and **log** signals and optional manual fills. **No automatic order placement** unless you explicitly change project scope in writing (same policy as Phase 1).

Phase 2 is **split into multiple plans** so each file stays focused and testable:

| Plan | File | What it covers |
|------|------|----------------|
| **A — Data & credits** | **[plan-phase2-01-data-ingest.md](plan-phase2-01-data-ingest.md)** | Incremental or interval ingest, session windows, cache strategy, credit discipline, tests. |
| **B — Inference & loop** | **[plan-phase2-02-inference-runtime.md](plan-phase2-02-inference-runtime.md)** | Load `model_bundle`, rebuild entry-time features for “now,” `predict_proba`, long-running tick loop, replay tests. |
| **C — State, UI, journal** | **[plan-phase2-03-state-ui-journal.md](plan-phase2-03-state-ui-journal.md)** | Position file, entry vs exit mode, stop/TP display vs model scores, structured logs, optional GUI, journal append. |

**Suggested order:** A → B → C. Plan B can start with **historical replay** before Plan A is finished, but **production** use should wait until A defines safe refresh behavior.

---

## Prerequisites (owner)

Before treating Phase 2 as “in scope” for implementation:

1. You are **satisfied enough** with **offline** model quality (train/val behavior, `sparkles report`, optional `journal compare`) that spending API credits on fresher data is justified. This mirrors **[METHODOLOGY.md](../METHODOLOGY.md)** historical-first policy.
2. You **explicitly approve** starting **Phase 2 — Plan A** (in chat or by marking the **Owner approval** line in that plan’s first iteration).

---

## Mandatory rules (Phase 2 — all plans)

These apply to **every** agent or contributor working on Phase 2.

1. **Approval gate:** Do **not** begin the **next** iteration **inside a given plan file** until the **owner approves** that next iteration (chat or checkbox / date in that file).
2. **Single-step focus:** Complete **at most one iteration** per owner request unless the owner explicitly asks to chain iterations.
3. **Progress log:** After substantive work, **append** one row to that plan file’s **Progress & change log (append-only)** table (bottom of the file): ISO date, short summary, paths touched, iteration id (e.g. **A2**).
4. **API credits:** Follow **`.cursor/rules/sparkles-api-credits.mdc`** and **[METHODOLOGY.md](../METHODOLOGY.md)** §2.4. No high-frequency hammering; every new fetch path needs **documented** default intervals and **tests** or dry-run instructions.
5. **No broker execution:** Recommendations and logs only, unless the owner changes scope in writing.
6. **Conflict resolution:** If Phase 2 notes conflict with **Phase 1** closure facts, **`plan.md`** and **`METHODOLOGY.md`** win unless you update them intentionally.

---

## How to use these plans (owner workflow)

1. Open **Plan A**, read **Iteration 1**, implement or delegate, run the **How you can test** steps, then approve **Iteration 2**.
2. Repeat per iteration. When Plan A’s iterations are done, move to Plan B, then C.
3. Keep **ML quality** work in **[ML_EXPANSION.md](ML_EXPANSION.md)** where possible (preprocessing, metrics); link from Phase 2 progress log when a training change was required for inference parity.

---

## Document map

| File | Role |
|------|------|
| **[plan.md](../plan.md)** | Phase 1 iterations and history (unchanged role). |
| **This file** | Phase 2 index and shared rules. |
| **[plan-phase2-01-data-ingest.md](plan-phase2-01-data-ingest.md)** | Data refresh / session windows / credits. |
| **[plan-phase2-02-inference-runtime.md](plan-phase2-02-inference-runtime.md)** | Scoring loop and bundle load. |
| **[plan-phase2-03-state-ui-journal.md](plan-phase2-03-state-ui-journal.md)** | Position, messaging, logging, journal. |
| **[METHODOLOGY.md](../METHODOLOGY.md)** | Product intent and API philosophy. |
| **[DEVELOPER.md](../DEVELOPER.md)** | Code map (update when new CLIs/modules land). |

---

## Progress & change log (append-only) — overview-level

**Instructions:** Add new rows **only below** this line, newest at the bottom. Use this table for **cross-cutting** Phase 2 milestones (optional). Each sub-plan also has its own log.

| Date (ISO) | Summary | Plan / iteration | Paths / notes |
|------------|---------|------------------|---------------|
| 2026-04-11 | Phase 2 plan set drafted: overview + Plans A–C with per-iteration owner tests. | Setup | `docs/plan-phase2-*.md`, `docs/README.md` |
| 2026-04-12 | Plan A **A1**: design block in `plan-phase2-01-data-ingest.md`, **`live_ingest`** validated config (default off). | A1 | `sparkles/config/schema.py`, `sparkles/reporting/summary.py`, `configs/experiments/rklb_baseline.yaml` |
