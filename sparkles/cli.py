"""CLI: ingest → label → train.

Stubs until Iterations 2–6. Always pass --config or run from repo root so the
default path resolves.
"""

from __future__ import annotations

from pathlib import Path

import typer

from sparkles.config import load_experiment_config

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
) -> None:
    """Download and cache 1m bars (Iteration 2)."""
    cfg_path = _resolve_config(config)
    _ = load_experiment_config(cfg_path)
    typer.echo(f"[ingest] config OK: {cfg_path} (Iteration 2)")


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
) -> None:
    """Build triple-barrier labels (Iteration 4)."""
    cfg_path = _resolve_config(config)
    _ = load_experiment_config(cfg_path)
    typer.echo(f"[label] config OK: {cfg_path} (Iteration 4)")


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
