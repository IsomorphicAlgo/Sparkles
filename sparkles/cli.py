"""CLI: ingest → label → train.

Stubs until Iterations 2–6. Always pass --config or run from repo root so the
default path resolves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from sparkles.config import load_experiment_config
from sparkles.data.ingest import run_ingest
from sparkles.labels.triple_barrier import run_label

app = typer.Typer(
    help="Sparkles swing ML pipeline (Phase 1). Use --config for experiment YAML.",
    no_args_is_help=True,
)

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
) -> None:
    """Train baseline model (Iteration 6)."""
    cfg_path = _resolve_config(config)
    _ = load_experiment_config(cfg_path)
    typer.echo(f"[train] config OK: {cfg_path} (Iteration 6)")


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
) -> None:
    """Summarize last run / artifacts (Iteration 7)."""
    cfg_path = _resolve_config(config)
    _ = load_experiment_config(cfg_path)
    typer.echo(f"[report] config OK: {cfg_path} (Iteration 7)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
