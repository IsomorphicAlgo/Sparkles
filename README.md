# Sparkles

**Sparkles** is a Python project for **single-stock** swing / intraday **research**: pull **1-minute** US equity history from **TwelveData**, cache it as **Parquet**, and (per the roadmap) label with **triple-barrier** methods, train a model, and respect a **conservative day-trade cap** for future advisory use.

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

The command prints the path to a **Parquet** file under `data/cache/`. See **[DEVELOPER.md](DEVELOPER.md)** for `--force`, cache TTL, and free-tier throttling.

## Documentation

| Doc | Contents |
|-----|----------|
| **[METHODOLOGY.md](METHODOLOGY.md)** | What we’re building: data flow, labeling idea, PDT cap, API credit discipline, roadmap stages. |
| **[DEVELOPER.md](DEVELOPER.md)** | Where to change symbol, training code, ingest settings. |
| **[plan.md](plan.md)** | Approval-gated iterations and append-only progress log. |

## Security

- Put your API key in the **environment** or a **local `.env`** (see `.env.example`). **Do not commit secrets.**
- `.gitignore` excludes `.env`, `data/cache/`, and `*.parquet` by default.

## CLI (roadmap)

```text
sparkles ingest   # historical 1m → Parquet (implemented)
sparkles label    # triple-barrier labels (planned)
sparkles train    # train + artifacts (planned)
sparkles report   # summaries (planned)
```

## License

Add a `LICENSE` file when you are ready to publish the repo publicly.
