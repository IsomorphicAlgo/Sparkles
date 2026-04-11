# Sparkles — methodology

This document describes **what** we are building and **why**, at a level suitable for you or a future contributor. **§9** is an expanded **how to run** guide (CLI, install options, train/report behavior). **§10** collects **tips and tricks** for training and parameter choices. Operational “where to edit” detail lives in **[DEVELOPER.md](DEVELOPER.md)**; the repo quick start is **[README.md](README.md)**. The **approval-gated roadmap** and history live in **[plan.md](plan.md)**.

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
- Paths for cache and artifacts (`paths.cache_dir`, `paths.artifacts_dir`).
- **`model:`** — classifier family and hyperparameters: `type` is **`logistic_regression`** (always available) or **`xgboost_classifier`** (requires optional install **`pip install -e ".[ml]"`**). Logistic options include `solver`, `tol`, `logistic_c`, `max_iter`, `class_weight`. XGBoost options include `xgb_n_estimators`, `xgb_max_depth`, `xgb_learning_rate`, and related knobs (see **`DEVELOPER.md`**).
- **`train:`** — minimum train/val row counts, val unseen-class handling, optional `experiment_name` / `notes`, and **`export_predictions`** (`val` / `all` / `none`) for **`predictions.parquet`** next to **`metrics.json`**.
- **`journal:`** (optional) — **`csv_path`** to a personal trade log for **`sparkles journal compare`** (see §9.5).
- **`features:`** — toggles for **entry-time** feature groups (e.g. label geometry, intraday range, volume). Turning a group off changes the feature matrix without editing Python; see **`DEVELOPER.md`** for the column map.

Details and file pointers: **`DEVELOPER.md`**. Broader ML roadmap (phases beyond Phase 1): **`docs/ML_EXPANSION.md`**.

---

## 4. Pipeline stages (Phase 1 roadmap)

Work proceeds in **iterations** documented in **`plan.md`**; the next stage starts only after **owner approval**.

| Stage | Role |
|--------|------|
| **Ingest** | TwelveData 1m historical → normalized OHLCV → Parquet cache. |
| **Volatility** | Daily-close log returns → rolling std over `vol_lookback_trading_days`, **`shift(1)`**, √252 annualization; broadcast to every 1m bar on that session date (`sparkles/features/volatility.py`). |
| **Labels (triple barrier)** | For each candidate entry time: upper barrier (take-profit path), lower barrier (stop), vertical barrier (max holding time). Moves scaled by **recent volatility** vs a reference; effective TP floored by **`min_profit_per_trade_pct`**. Path uses **full 1m forward path**, including **same-day** touches (labels match intraday reality). |
| **Day-trade ledger** | Rolling **weekday** window (`rolling_business_days`), **≤ `max_day_trades`** day-trade **events**; for backtests and **future** simulation/advisory logic (holidays not excluded in v1). |
| **Features + train** | Entry-only **`features:`** join; session-date split; **`model.type`** via **`estimators.py`**; **`sparkles train`** → bundle + **`metrics.json`** + optional **`predictions.parquet`** + **`experiments.jsonl`**. |
| **Closure** | **`sparkles report`** (cache paths, full YAML parameter summary, latest **`metrics.json`**, **`experiments.jsonl`** tail); train prints headline metrics; formal Phase 1 sign-off in **`plan.md`**. |

---

## 5. Machine-learning framing

- **Supervised learning** on **triple-barrier outcomes** (e.g. which barrier hit first, or derived binary/ternary targets).
- **Validation**: time-based / walk-forward splits—not random row splits—for series data.
- **Orchestration** lives in **`sparkles/models/train.py`** (load → features → split → fit → save). **Estimator choice and factory** live in **`sparkles/models/estimators.py`**; stable hyperparameters belong in YAML under **`model:`** (and **`features:`**, **`train:`**) so runs stay reproducible and comparable.

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
| **METHODOLOGY.md** | This file — concepts, methodology, **how to run** (§9), **tips** (§10), **label stride** (§11), **metrics & class_weight** (§12). |
| **DEVELOPER.md** | Where to edit symbol, training file, ingest knobs. |
| **plan.md** | Iterations, approvals, append-only progress log. |
| **[docs/README.md](docs/README.md)** | Index of files under `docs/`. |
| **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)** | Post–Phase 1 roadmap: models, features, YAML expansion (append-only log). |

---

## 9. How to run the program (terminal)

From the **repository root**, with your experiment YAML (default path used below). Set **`TWELVEDATA_API_KEY`** first (environment or `.env` per **`.env.example`**).

**Windows PowerShell (API key for the current session):**

```powershell
$env:TWELVEDATA_API_KEY = "your_key_here"
cd "C:\path\to\Sparkles"
```

### 9.1 Install

**Core + developer tools** (lint, types, tests):

```bash
python -m pip install -e ".[dev]"
```

**Optional XGBoost** (only if `model.type: xgboost_classifier` in YAML):

```bash
python -m pip install -e ".[dev,ml]"
```

### 9.2 Data and labels

**Historical ingest** — downloads the configured `data_start`–`data_end` window, writes 1m Parquet under `paths.cache_dir` (default `data/cache/`), prints the **absolute path** to the file:

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml
```

Verbose chunk logging; **force** a full re-download (ignores cache age; uses API credits):

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
sparkles ingest -c configs/experiments/rklb_baseline.yaml --force -v
```

**Triple-barrier labels** — needs the ingest Parquet for the same `symbol`, `data_start`, and `data_end`. Writes labeled Parquet, prints path and **`barrier_outcome`** value counts:

```bash
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles label -c configs/experiments/rklb_baseline.yaml -v
```

### 9.3 Risk (day-trade cap)

Dry-run using **`max_day_trades`** and **`rolling_business_days`** from the YAML (ledger is for future backtests / advisory; not applied inside the labeler today):

```bash
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml --as-of 2026-04-01
sparkles risk day-trades -c configs/experiments/rklb_baseline.yaml --as-of 2026-04-01 --history 2026-03-25,2026-03-26
```

### 9.4 Train

**Prerequisites:** labeled Parquet exists, and **`train_start` / `train_end` / `val_start` / `val_end`** are set in the YAML.

**What it does:** builds the feature matrix from **`features:`**, time-splits by **US session date**, fits **`model.type`** (`logistic_regression` or `xgboost_classifier` with **`[ml]`**), writes **`artifacts/{SYMBOL}/{run_id}/`**: **`model_bundle.joblib`**, **`metrics.json`**, **`experiment_config.json`** (full experiment snapshot, JSON-serializable), and (unless **`train.export_predictions: none`**) **`predictions.parquet`** with per-row **`entry_time`**, **`session_date`**, **`split`**, **`y_true`**, **`y_pred`**, and probability columns when supported. Appends one line to **`artifacts/experiments.jsonl`** including the same snapshot under **`experiment_config`** plus headline metrics.

```bash
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml -v
```

On success, the CLI prints the **artifact directory**, **`model_type`**, train/val accuracy, and row counts.

**Training log (CSV):** flatten **`experiments.jsonl`** for spreadsheets (default: rows for the YAML symbol only; use **`--all-symbols`** for the entire log):

```bash
sparkles experiments export -c configs/experiments/rklb_baseline.yaml
sparkles experiments export -c configs/experiments/rklb_baseline.yaml -o out/runs.csv
sparkles experiments export -c configs/experiments/rklb_baseline.yaml --all-symbols
```

### 9.5 Journal compare (optional)

If **`journal.csv_path`** is set in YAML to a CSV of **your** trades, **`sparkles journal compare`** left-joins that file to **aggregated** model predictions by **entry session date**. Use this to line up “what I did on day X” with “what the model saw on labeled entries that day.” **Long holds** are fine: one journal row per **open**, with **`exit_date`** months later if you like; alignment uses **entry date** only (not daily P&L over the hold). See **`data/journal/README.md`** and **`configs/examples/journal_trades.example.csv`**.

```bash
sparkles journal compare -c configs/experiments/rklb_baseline.yaml
sparkles journal compare -c configs/experiments/rklb_baseline.yaml --run 20260411T015314_621888Z
sparkles journal compare -c configs/experiments/rklb_baseline.yaml --split val
```

**`--split`** controls which prediction rows are rolled up before the join: **`val`** (default), **`train`**, or **`both`**. Output: **`journal_compare.csv`** in the chosen run directory.

### 9.6 Report (smoke and audit)

**`sparkles report`** prints a structured summary in one go:

- Whether **ingest** and **labeled** Parquet paths exist (and resolved paths).
- **Parameters from the current YAML** — splits, labeling knobs, ingest throttling headline, **`model:`**, **`train:`**, and compact **`features:`** JSON.
- **Latest training run** (or the run given by **`--run <run_id>`**): headline **`metrics.json`** lines including **`model_type`**, accuracies, **`classes`**, and the **`features`** dict **as stored at train time** (useful if you edited YAML after training).
- **Tail of `experiments.jsonl`** for that symbol — each line includes run id, val accuracy, model type/solver (or XGBoost marker), class weight, feature flags, optional experiment name/notes.

```bash
sparkles report -c configs/experiments/rklb_baseline.yaml
sparkles report -c configs/experiments/rklb_baseline.yaml --run 20260411T015314_621888Z
```

### 9.7 Typical end-to-end order

Use the **same `--config`** path for every step:

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles report -c configs/experiments/rklb_baseline.yaml
```

### 9.8 Help

```bash
sparkles --help
sparkles ingest --help
sparkles label --help
sparkles risk day-trades --help
sparkles train --help
sparkles journal compare --help
sparkles experiments export --help
sparkles report --help
```

---

## 10. Tips and tricks (training and parameters)

These are practical reminders aligned with what Sparkles is trying to achieve: **honest offline research**, **no leakage**, and **runs you can compare**.

- **Separate “label world” from “model world.”** Changing barriers, `min_profit_per_trade_pct`, `label_entry_stride`, vol lookback, or the ingest date range changes **labels** (and usually requires **`label`** again, and possibly **`ingest`**). Changing only **`model:`**, **`train:`**, or **`features:`** typically needs **`train`** (and **`report`**) only. Mixing the two without re-running steps is a common source of confusion.

- **Trust the time split, not random shuffles.** Train and validation are separated by **calendar session dates** in the exchange timezone. Do not evaluate with a random row split on the same file; it will overstate quality on sequential market data.

- **Watch class balance.** Triple-barrier outcomes are often **imbalanced**. Use **`report`** and **`metrics.json`** (and outcome counts from **`label`**) to see whether accuracy is meaningful. YAML **`model.class_weight`** helps logistic regression; for **XGBoost**, the same setting is translated into **per-row sample weights** on fit.

- **YAML vs artifacts.** After you train, **`metrics.json`** and **`experiments.jsonl`** record **`model_type`** and **`features`** as they were **at train time**. **`report`** shows both the **current YAML** and the **stored metrics** so you can spot drift (e.g. you changed `features:` but did not retrain).

- **Feature toggles are causal, not cosmetic.** Disabled groups remove columns from **X**; keep at least one group enabled. All enabled columns still use only information **at the entry bar** (no future path in **X**).

- **API credits when iterating on data.** Re-running **`ingest --force`** on a long window burns **TwelveData** credits. Prefer **`train`** / **`report`** while tuning model and feature YAML on a fixed cache when possible.

- **XGBoost vs logistic.** Logistic regression is **fast, interpretable, and dependency-light**. XGBoost can capture nonlinearity but needs **`[ml]`**, tuning, and care not to **overfit** small or noisy regimes—compare runs using the same splits and check val behavior, not only train accuracy.

- **`train.drop_val_unseen_classes`.** If the validation period contains **outcome labels never seen in train**, the default is to **drop** those val rows (with a log warning). If you set this to **`false`**, training **fails** when that situation occurs—useful to force yourself to fix date ranges or class coverage instead of silently skewing metrics.

- **Ledger vs labels.** The **day-trade ledger** encodes a **3-in-5** style cap for **future** simulation or advice. **Triple-barrier labels** still use the **full intraday path** for the mechanistic outcome. Do not assume the ledger “fixed” label semantics unless you build a separate labeling mode for that.

- **Journal vs model targets.** **`journal compare`** merges your **real** entries to **triple-barrier** prediction rows by **date**; it does not equate your hold horizon (e.g. six months) with the label horizon. Use it as a qualitative alignment tool, not a guarantee that the model was trained to replicate your style.

---

## 11. Label entry stride (what we are doing and why)

**`label_entry_stride`** in the experiment YAML controls **how densely** we place hypothetical long entries on the 1m grid: bar indices **`0, N, 2N, …`**. It is **explicit** in **`configs/experiments/rklb_baseline.yaml`** so runs are reproducible and the labeled Parquet name (`…_s{N}.parquet`) matches config.

**Default posture (`390`):** About **one candidate entry per regular US session**—a coarse view aligned with “not deciding every minute,” cheaper **`sparkles label`**, and fewer **highly redundant** adjacent rows for training.

**Dense posture (`1`):** **Every minute** is a candidate—richer coverage of intraday paths where a **percentage move can finish inside a session**, at the cost of **much larger** label files, **longer** labeling time, and **strong correlation** between neighboring rows (not independent samples).

**What we hope to accomplish:** Keep **one clear knob** to move between **session-level** and **minute-level** research questions without code changes, while documenting that **PDT / day-trade limits** apply to **execution policy**, not to how many bars we label. Iterating on stride requires **`sparkles label`** again (then **`train`**); see §10 “label world vs model world.”

---

## 12. Where to read `classification_report_val` and how `model.class_weight` works

### 12.1 `classification_report_val` (validation, per class)

After **`sparkles train`**, each run writes **`metrics.json`** under:

**`artifacts/{SYMBOL}/{run_id}/metrics.json`**

(The CLI prints the resolved **`run_id`** folder path when training finishes.)

Inside that JSON, the key **`classification_report_val`** holds the **validation** split report from scikit-learn: for each **`barrier_outcome`** class seen in the encoder, **`precision`**, **`recall`**, **`f1-score`**, and **`support`** (row count). It also includes **`accuracy`**, **`macro avg`**, and **`weighted avg`** over classes.

**Ways to view it:**

- Open **`metrics.json`** in an editor and search for **`classification_report_val`**.
- **PowerShell** (from repo root, adjust run folder name):

```powershell
Get-Content "artifacts\RKLB\YOUR_RUN_ID\metrics.json" -Raw | ConvertFrom-Json | Select-Object -ExpandProperty classification_report_val | ConvertTo-Json -Depth 6
```

- **Python:**

```python
import json
from pathlib import Path
m = json.loads(Path("artifacts/RKLB/YOUR_RUN_ID/metrics.json").read_text(encoding="utf-8"))
print(json.dumps(m["classification_report_val"], indent=2))
```

**`sparkles report`** summarizes the **latest** run’s headline numbers; for **full per-class** validation detail, use **`metrics.json`** as above.

### 12.2 `model.class_weight` in experiment YAML

Configured under **`model:`** in **`configs/experiments/*.yaml`**, validated by **`sparkles/config/schema.py`**. Sparkles allows **three** shapes (this is slightly **stricter** than typing every sklearn string by hand):

| YAML form | Meaning |
|-----------|---------|
| **Omitted** or **`null`** | No class weighting (**`None`** in sklearn): all training rows weighted equally. |
| **`balanced`** | sklearn’s **`balanced`** mode: weights are adjusted inversely to class frequency in the **training** labels (helps minority classes like **`take_profit`** / **`vertical`**). |
| **Mapping (object)** | **Per-class weights**: keys are **`barrier_outcome`** strings (**`stop_loss`**, **`take_profit`**, **`vertical`**, **`end_of_data`** as applicable), values are positive floats. Example: boost rare classes manually. |

**Not supported as a bare string** in YAML (besides **`balanced`**): e.g. sklearn’s historical **`balanced_subsample`** for ensembles is **not** wired for **`logistic_regression`** here—use **`balanced`** or a **dict**.

For **`xgboost_classifier`**, the same **`model.class_weight`** is translated into **per-row sample weights** on **`fit`** (see **`sparkles/models/estimators.py`**): **`null`** → uniform weights, **`balanced`** → sklearn **`compute_sample_weight`**, **dict** → weights by class name then expanded per row.

**Tip:** If validation shows **0 recall** on rare classes with **`class_weight` omitted**, try **`class_weight: balanced`** before hand-tuning a dict; dicts are for when you want **explicit** relative costs (e.g. weight **`take_profit`** higher than **`vertical`**).
