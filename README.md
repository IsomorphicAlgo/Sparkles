# Sparkles

## Purpose and general use

**Today:** Sparkles is an **offline research** toolkit for **one US ticker at a time**. It pulls **historical** 1-minute bars from **TwelveData**, caches them as **Parquet**, builds **triple-barrier labels** (mechanistic outcomes after a hypothetical long: take-profit, stop-loss, time exit, end-of-data), and **trains classifiers** on those labels. Labels are **training targets**, not buy/sell signals.

**Configuration:** One **YAML experiment file** (see `configs/experiments/`, validated by Pydantic) drives symbol, date ranges, barriers, ingest throttling, **train/val session dates**, **`model:`** (e.g. **logistic regression** or **XGBoost** with an optional extra), **`train:`** (row floors, val class handling, run notes), and **`features:`** (entry-time feature groups you can turn on or off without editing Python).

**Models:** Trained artifacts are meant to **support** future tooling (e.g. comparing a setup to past labeled paths). They do **not** replace your judgment. This repo has **no broker integration** and **no auto-trading**.

**Later (only if you expand scope in writing):** Optional **live or periodic** refresh, trade logging, and **recommendations** you confirm yourself. Until then the stack stays **batch historical**—no always-on polling— to protect **TwelveData API credits** and keep the research loop simple.

**Risk:** A **day-trade ledger** (default **≤ 3 day trades / 5 US business days**) supports staying under typical **PDT**-style pattern limits in **future** backtests or advisory flows; it does not change how triple-barrier labels are computed today.

---

The CLI ties it together: **ingest** → **label** → **train** → **report**, plus **risk** dry-runs. **`sparkles report`** summarizes cache paths, **current YAML** parameters (including model, train, and features), the latest **`metrics.json`** (including **`model_type`** and stored feature flags), and recent **`experiments.jsonl`** rows.

## Quick start

**Requirements:** Python 3.10+, a [TwelveData](https://twelvedata.com/) API key.

```bash
cd Sparkles
python -m pip install -e ".[dev]"
```

For **`model.type: xgboost_classifier`** in YAML, install XGBoost as well:

```bash
python -m pip install -e ".[dev,ml]"
```

**Windows — set the API key** (pick one shell):

```bat
set TWELVEDATA_API_KEY=your_key_here
```

```powershell
$env:TWELVEDATA_API_KEY = "your_key_here"
```

**Minimal pipeline** (same `--config` for every step; set `train_*` / `val_*` in the YAML before `train`):

```bash
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
sparkles label -c configs/experiments/rklb_baseline.yaml
sparkles train -c configs/experiments/rklb_baseline.yaml
sparkles report -c configs/experiments/rklb_baseline.yaml
```

- **Ingest** prints the absolute path to the cached 1m Parquet under `data/cache/` (or `paths.cache_dir`).
- **Train** prints the run directory under `artifacts/{SYMBOL}/` plus headline metrics (including `model_type`).
- **Report** is the smoke check: paths, YAML snapshot, latest metrics, experiments tail. Optional: `sparkles report -c … --run <run_id>`.

**Deeper detail:** **[DEVELOPER.md](DEVELOPER.md)** (where to edit), **[METHODOLOGY.md](METHODOLOGY.md)** (why, how to run §9, tips §10), **[plan.md](plan.md)** (Phase 1 history and approvals).

## Documentation

| Doc | Contents |
|-----|----------|
| **[METHODOLOGY.md](METHODOLOGY.md)** | Product intent, data and API credit philosophy, **how to run** (§9), **training tips** (§10). |
| **[DEVELOPER.md](DEVELOPER.md)** | Repo layout, YAML knobs, features/labels/train paths, CLI smoke order. |
| **[plan.md](plan.md)** | Approval-gated Phase 1 iterations and append-only progress log. |
| **[docs/README.md](docs/README.md)** | Index of files under `docs/`. |
| **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)** | Post–Phase 1 ML roadmap (preprocessing, dry-run, richer metrics, etc.). |

## Security and local data

- Put **`TWELVEDATA_API_KEY`** in the **environment** or a **local `.env`** (see **`.env.example`**). Do **not** commit secrets.
- **`.gitignore`** excludes env files (with **`!.env.example`** kept), common credential filenames, **`data/cache/`**, **`data/journal/*.csv`** (personal trade logs), **`artifacts/`**, **`*.parquet`**, Python and tool caches, and typical IDE/OS junk. See the file at the repo root for the full list.

## CLI

```text
sparkles ingest              # historical 1m → Parquet (cache-first, throttled)
sparkles label               # triple-barrier labels → Parquet + outcome counts
sparkles risk day-trades     # day-trade cap dry-run from YAML (+ optional history)
sparkles train               # fit model; bundle + metrics + predictions.parquet (default val)
sparkles journal compare     # join journal CSV to predictions by entry date (optional)
sparkles report              # paths + YAML params + latest metrics + experiments tail
```

```bash
sparkles --help
sparkles ingest --help
sparkles label --help
sparkles risk day-trades --help
sparkles train --help
sparkles journal compare --help
sparkles report --help
```

## License

Add a `LICENSE` file when you are ready to publish the repo publicly.
