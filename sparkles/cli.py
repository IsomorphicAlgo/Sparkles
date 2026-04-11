"""CLI: ingest → label → risk → train → report.

Pass ``--config`` or run from repo root so the default experiment YAML resolves.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from sparkles.config import load_experiment_config
from sparkles.data.ingest import run_ingest
from sparkles.labels.triple_barrier import run_label
from sparkles.models.train import run_train
from sparkles.reporting.summary import run_phase1_report
from sparkles.risk.day_trade_ledger import DayTradeLedger

app = typer.Typer(
    help="Sparkles swing ML pipeline (Phase 1). Use --config for experiment YAML.",
    no_args_is_help=True,
)

risk_app = typer.Typer(
    help="Risk checks (e.g. rolling day-trade cap vs config).",
    no_args_is_help=True,
)
app.add_typer(risk_app, name="risk")

_DEFAULT_CONFIG = Path("configs/experiments/rklb_baseline.yaml")


def _resolve_config(config: Path | None) -> Path:
    if config is not None:
        return config
    if _DEFAULT_CONFIG.is_file():
        return _DEFAULT_CONFIG
    raise typer.BadParameter(
        "Pass --config PATH or create configs/experiments/rklb_baseline.yaml",
    )


@app.command()
def ingest(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (default: configs/experiments/rklb_baseline.yaml)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Ignore cache TTL and re-download the full range",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Log each chunk to stderr",
    ),
) -> None:
    """Download and cache 1m bars (historical batch; TwelveData)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    try:
        path = run_ingest(cfg, force_refresh=force)
    except ValueError as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    typer.echo(str(path.resolve()))


@app.command()
def label(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Progress logging",
    ),
) -> None:
    """Build triple-barrier labels from cached 1m Parquet; write labeled Parquet."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    try:
        out = run_label(cfg)
    except (FileNotFoundError, KeyError, ValueError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    typer.echo(str(out.resolve()))
    summary = pd.read_parquet(out)["barrier_outcome"].value_counts()
    typer.echo(summary.to_string())


def _parse_history_dates(history: str) -> list[date]:
    out: list[date] = []
    for part in history.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(date.fromisoformat(p))
    return out


@risk_app.command("day-trades")
def risk_day_trades(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (max_day_trades, rolling_business_days)",
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Decision date (ISO). Default: today (local).",
    ),
    history: str = typer.Option(
        "",
        "--history",
        help="Comma-separated ISO dates (one day-trade event per token)",
    ),
) -> None:
    """Dry-run: count day trades in rolling window; show whether one more is allowed."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    ledger = DayTradeLedger(
        max_day_trades=cfg.max_day_trades,
        rolling_business_days=cfg.rolling_business_days,
    )
    for d in _parse_history_dates(history):
        ledger.record(d)
    anchor = date.fromisoformat(as_of) if as_of else date.today()
    n = ledger.count_in_window(anchor)
    ok = ledger.can_add_day_trade(anchor)
    typer.echo(
        f"as_of={anchor.isoformat()}  "
        f"window_business_days={cfg.rolling_business_days}  "
        f"events_in_window={n}  max={cfg.max_day_trades}  "
        f"can_add_day_trade={ok}",
    )


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Progress logging",
    ),
) -> None:
    """Fit baseline classifier; write ``model_bundle.joblib`` + ``metrics.json``."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    try:
        out = run_train(cfg)
    except (FileNotFoundError, ValueError, KeyError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    typer.echo(str(out.resolve()))
    metrics_path = out / "metrics.json"
    if metrics_path.is_file():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        typer.echo(
            f"train_accuracy={m['train_accuracy']:.4f}  "
            f"val_accuracy={m['val_accuracy']:.4f}  "
            f"train_n={m['train_n']}  val_n={m['val_n']}",
        )


@app.command()
def report(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run",
        help="Subfolder under artifacts/SYMBOL/ (default: latest run with metrics)",
    ),
) -> None:
    """Show cache paths, latest train metrics, and recent experiments.jsonl rows."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    typer.echo(run_phase1_report(cfg, run_id=run_id))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
