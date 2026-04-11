# Phase 2 — Plan B: Inference runtime and scoring loop

**Parent index:** **[plan-phase2-overview.md](plan-phase2-overview.md)**.

**Depends on:** Plan A for **fresh bars** in production; you can still implement **B1–B3** using **historical Parquet only** (replay mode).

**Goal:** Given a **trained** `model_bundle.joblib` and experiment YAML, compute **the same entry-time feature vector** the trainer used for **the latest** (or a chosen) bar, run **`predict_proba`** (when available), and emit a **stable JSON or structured record** per tick. Later iterations add a **loop** that repeats on new data.

**Out of scope for Plan B:** Position state and exit policy (Plan C); broker connectivity.

---

## Mandatory rules

Follow **[plan-phase2-overview.md](plan-phase2-overview.md)** — Mandatory rules (Phase 2 — all plans).

---

## Iteration B1 — Bundle load and config snapshot parity

- **Goal:** Document and implement **one** supported load path: `model_bundle.joblib` + YAML **must agree** on `features:` flags and symbol semantics (or bundle carries a frozen feature list and code validates YAML against it).
- **Deliverables:** Loader helper (e.g. `sparkles/models/bundle_load.py`), clear error when feature flags **mismatch** bundle metadata, tests with a tiny saved bundle fixture.
- **Done when:** Loading a known run’s bundle and passing the **same** YAML used at train time raises **no** warning; wrong YAML fails loudly.

### How you can test (owner)

1. Train once: `sparkles train -c configs/experiments/your.yaml` and note the printed **run id**.
2. Run the new **verify** command or Python snippet from **DEVELOPER.md** (to be added) pointing at that run and the **same** YAML; expect **OK**.
3. Edit YAML `features:` (toggle one flag), run again; expect **error** explaining mismatch.

**Owner approval to proceed to B2:** `[ ]` Date: ___________

---

## Iteration B2 — Single-bar feature build + prediction (replay)

- **Goal:** For a **single** timestamp row (or latest row in a DataFrame slice), build **X** using existing `build_feature_matrix` / registry paths, run **prediction**, output **classes + probabilities** to stdout or a JSON file.
- **Deliverables:** Function `score_entry_at(...)` (name flexible) with tests; **no** lookahead: features must match Phase 1 **entry-only** contract (**[DEVELOPER.md](../DEVELOPER.md)**).
- **Done when:** On a frozen historical snippet, scores match **`predictions.parquet`** for the same `entry_time` row (within floating tolerance).

### How you can test (owner)

1. Pick a run with `train.export_predictions: all` so **`predictions.parquet`** exists.
2. Choose one **`entry_time`** from that file; run the new CLI, e.g. `sparkles score once -c … --run <id> --at <iso-timestamp>`.
3. Compare printed **`y_pred` / `proba_*`** to the Parquet row; they should **match**.

**Owner approval to proceed to B3:** `[ ]` Date: ___________

---

## Iteration B3 — Replay loop (offline “fake live”)

- **Goal:** Step through historical bars every **N** seconds (or as fast as possible) emitting the same structured record as B2, to **exercise** logging and downstream consumers **without** API calls.
- **Deliverables:** CLI `sparkles score replay -c … --parquet path --speed 10x` (shape is indicative), tests on tiny Parquet.
- **Done when:** Full replay completes with deterministic line count equals number of scored steps (document stride behavior).

### How you can test (owner)

1. Run replay on a **small** cached file (one session); redirect stdout to a file; confirm **one line per bar** (or per stride) and no exceptions.
2. Compare **first** and **last** replay lines to manually selected bars.

**Owner approval to proceed to B4:** `[ ]` Date: ___________

---

## Iteration B4 — Tie-in to refreshed data (Plan A)

- **Goal:** After each **successful** refresh (Plan A), run **B2** on the **latest** bar automatically in a **single** command path, e.g. `sparkles live tick -c …` that: refresh once → score once → exit (for cron) **or** optionally chains into the refresh loop.
- **Deliverables:** Composition only—minimal new logic; document ordering (volatility columns must exist on the frame used for scoring).
- **Done when:** One command produces a **timestamped score** after refresh.

### How you can test (owner)

1. With Plan A refresh working, run `sparkles live tick` (or agreed name) **once**; confirm log shows **refresh** then **score**.
2. Disconnect network (or revoke key temporarily): confirm **clear failure** and **no** silent score on stale data unless explicitly overridden by flag.

**Owner approval to proceed to B5:** `[ ]` Date: ___________

---

## Iteration B5 — “Risk rating” surface (optional but aligned with your vision)

- **Goal:** Without claiming causal explanations, expose a **single** derived **score** (e.g. max class probability, margin between top-2 classes, or a simple mapping table in YAML). This is **presentation**, not a new model.
- **Deliverables:** Optional YAML thresholds: `advisory.entry.min_proba_take`, etc.; documented semantics.
- **Done when:** JSON line includes **`rating` or `signal_strength`** fields tested in unit tests.

### How you can test (owner)

1. Run **replay** with thresholds set; count how often **`entry_interesting`** (or your chosen label) flips—tune until behavior feels sensible on **past** data.
2. Confirm changing thresholds in YAML **does not** require retraining (only rescoring).

**Owner approval (Plan B complete):** `[ ]` Date: ___________

---

## Progress & change log (append-only)

**Instructions:** Append **only** below; newest at bottom.

| Date (ISO) | Summary | Iteration | Paths / notes |
|------------|---------|-----------|---------------|
| 2026-04-11 | Plan B drafted: B1–B5 with owner test steps. | — | `docs/plan-phase2-02-inference-runtime.md` |
