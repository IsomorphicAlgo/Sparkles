# Sparkles

## Purpose and general use

**Today (Phase 1):** Sparkles is an **offline** toolkit for **one ticker at a time**: **historical** 1m data from **TwelveData** → **Parquet** → **triple-barrier labels** → **train** classifiers. Labels describe which mechanistic rule would have fired first after a hypothetical long (take-profit, stop-loss, time exit, etc.)—they are **training targets**, not “buy/sell” judgments.

**Models:** The trained model is meant to **support** your decisions in a **future** program (e.g. scanning setups, suggesting entries/exits, honoring stops). It does **not** replace your judgment and is **not** wired to brokers in this repo.

**Later (when you explicitly expand scope):** You can add **live or periodic** data refresh, **logging** when you take or skip trades, and an assistant that **recommends** actions. **No auto-execution** unless you change that in writing. Until then, the project stays **batch historical** (no always-on API polling) to save credits and focus on research quality.

**Risk:** A configurable **day-trade ledger** (≤ 3 day trades / 5 US business days by default) is for **backtests and future advisory** logic, aligned with staying under typical **PDT** pattern limits.

---

**Sparkles** packages the above: **TwelveData** ingest, **Parquet** cache, **triple-barrier** labeling, **training**, **day-trade** checks, and **`sparkles report`** for a quick artifact summary.

**There is no broker connection and no auto-trading**—only data, ML, and (later) optional recommendations you act on yourself.

## Quick start

**Requirements:** Python 3.10+, a [TwelveData](https://twelvedata.com/) API key.

```bash
cd Sparkles
python -m pip install -e ".[dev]"
set TWELVEDATA_API_KEY=your_key_here          # Windows CMD
# $env:TWELVEDATA_API_KEY = "your_key_here"   # Windows PowerShell
sparkles ingest -c configs/experiments/rklb_baseline.yaml -v
```

The command prints the path to a **Parquet** file under `data/cache/`. Then run **`sparkles label`** and **`sparkles train`** with the same `--config` (after `train_*` / `val_*` dates are set in YAML). Use **`sparkles report -c …`** to print ingest/label paths, latest **`metrics.json`** summary, and recent **`experiments.jsonl`** lines.

**Phase 1 smoke (same config end-to-end):** `ingest` → `label` → `train` → `report`. Full knobs and paths: **[DEVELOPER.md](DEVELOPER.md)** (also **METHODOLOGY.md**, **plan.md**).

## Documentation

| Doc | Contents |
|-----|----------|
| **[METHODOLOGY.md](METHODOLOGY.md)** | What we’re building: data flow, labeling idea, PDT cap, API credit discipline, roadmap stages. |
| **[DEVELOPER.md](DEVELOPER.md)** | Where to change symbol, training code, ingest settings, smoke path. |
| **[plan.md](plan.md)** | Approval-gated iterations and append-only progress log. |
| **[docs/README.md](docs/README.md)** | Index of everything under `docs/` (e.g. ML expansion plan). |
| **[docs/ML_EXPANSION.md](docs/ML_EXPANSION.md)** | Living roadmap for richer models, features, and YAML experiments. |

## Security

- Put your API key in the **environment** or a **local `.env`** (see `.env.example`). **Do not commit secrets.**
- `.gitignore` excludes `.env`, `data/cache/`, and `*.parquet` by default.

## CLI

```text
sparkles ingest              # historical 1m → Parquet
sparkles label               # triple-barrier labels → Parquet + outcome counts
sparkles risk day-trades     # day-trade cap dry-run (config + optional history)
sparkles train               # sklearn baseline → artifacts + experiments.jsonl
sparkles report                # cache paths + latest metrics + experiments tail
```

## License

Add a `LICENSE` file when you are ready to publish the repo publicly.
