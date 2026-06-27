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

from sparkles.backtest.meta_label import (
    compare_entry_policies,
    format_compare_report,
    resolve_primary_run_dir,
    train_meta_label,
)
from sparkles.backtest.threshold_sweep import (
    format_sweep_report,
    run_threshold_sweep,
)
from sparkles.backtest.val_backtest import (
    format_backtest_report,
    resolve_run_dir,
    run_val_backtest,
)
from sparkles.config import load_experiment_config
from sparkles.data.ingest import run_ingest
from sparkles.journal.compare import run_journal_compare
from sparkles.labels.triple_barrier import run_label
from sparkles.models.train import dry_run_train, format_dry_run_report, run_train
from sparkles.reporting.summary import run_phase1_report
from sparkles.risk.day_trade_ledger import DayTradeLedger
from sparkles.tracking.experiments_csv import (
    experiments_log_path,
    export_experiments_to_csv,
)

app = typer.Typer(
    help="Sparkles swing ML pipeline (Phase 1). Use --config for experiment YAML.",
    no_args_is_help=True,
)

risk_app = typer.Typer(
    help="Risk checks (e.g. rolling day-trade cap vs config).",
    no_args_is_help=True,
)
app.add_typer(risk_app, name="risk")

journal_app = typer.Typer(
    help="Optional personal trade journal vs model predictions.",
    no_args_is_help=True,
)
app.add_typer(journal_app, name="journal")

experiments_app = typer.Typer(
    help="Training experiment log (experiments.jsonl) utilities.",
    no_args_is_help=True,
)
app.add_typer(experiments_app, name="experiments")

meta_label_app = typer.Typer(
    help="Phase I3 meta-label spike (secondary filter on primary take_profit signals).",
    no_args_is_help=True,
)
app.add_typer(meta_label_app, name="meta-label")

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
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Ticker to download (default: experiment symbol, e.g. RKLB)",
    ),
    interval: str | None = typer.Option(
        None,
        "--interval",
        "-i",
        help="TwelveData interval: 1min or 1day (default: 1min for main symbol; "
        "inferred from context_ingest for others)",
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
    """Download and cache OHLCV for one symbol/interval (historical batch; TwelveData).

    Uses data_start/data_end from the experiment YAML. One append-friendly Parquet
    per symbol/interval; extending data_end fetches only missing days (not the
    full history). Each symbol is cached independently:

      sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s SPY -i 1min
      sparkles ingest -c configs/experiments/rklb_daytrade_v2.yaml -s VIXY -i 1day
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    try:
        path = run_ingest(cfg, symbol=symbol, interval=interval, force_refresh=force)
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print row counts, class balance, and feature list; do not fit or save",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Progress logging",
    ),
) -> None:
    """Fit classifier; write bundle, metrics, experiment_config.json, predictions."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    if dry_run:
        report = dry_run_train(cfg)
        typer.echo(format_dry_run_report(report))
        if not report.ready:
            raise typer.Exit(code=1)
        return
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
            f"model_type={m.get('model_type', '?')}  "
            f"train_accuracy={m['train_accuracy']:.4f}  "
            f"val_accuracy={m['val_accuracy']:.4f}  "
            f"val_f1_macro={m.get('val_f1_macro', 0.0):.4f}  "
            f"val_f1_weighted={m.get('val_f1_weighted', 0.0):.4f}  "
            f"train_n={m['train_n']}  val_n={m['val_n']}",
        )


@journal_app.command("compare")
def journal_compare(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (journal.csv_path + symbol)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run",
        help="Run id under artifacts/SYMBOL/ (default: latest with predictions)",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="Aggregate predictions for this split: val, train, or both",
    ),
) -> None:
    """Join journal CSV to predictions; write journal_compare.csv in the run folder."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    root = Path.cwd()
    sym_dir = root / cfg.paths.artifacts_dir / cfg.symbol.upper()
    if not sym_dir.is_dir():
        typer.secho(
            f"No artifact directory for symbol: {sym_dir}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    run_dir: Path | None = None
    if run_id:
        cand = sym_dir / run_id
        if (cand / "predictions.parquet").is_file():
            run_dir = cand
        else:
            typer.secho(
                f"No predictions.parquet in {cand}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:
        subdirs = [p for p in sym_dir.iterdir() if p.is_dir()]
        subdirs.sort(key=lambda x: x.name, reverse=True)
        for p in subdirs:
            if (p / "predictions.parquet").is_file():
                run_dir = p
                break
        if run_dir is None:
            typer.secho(
                "No run with predictions.parquet found; train with "
                "train.export_predictions: val or all.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    split_filter: str | None
    if split == "both":
        split_filter = None
    elif split in ("val", "train"):
        split_filter = split
    else:
        typer.secho(
            "--split must be val, train, or both",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        merged, out_csv = run_journal_compare(
            cfg,
            run_dir,
            split_filter=split_filter,
            base_dir=root,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    typer.echo(str(out_csv.resolve()))
    n_match = (
        int(merged["model_matched"].sum())
        if len(merged) and "model_matched" in merged.columns
        else 0
    )
    typer.echo(f"rows={len(merged)}  matched={n_match}")
    if len(merged):
        typer.echo(merged.head(12).to_string(index=False))


@experiments_app.command("export")
def experiments_export(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (for paths.artifacts_dir and default symbol filter)",
    ),
    output: Path = typer.Option(
        Path("artifacts/training_log.csv"),
        "--output",
        "-o",
        help="Output CSV path (default: artifacts/training_log.csv)",
    ),
    all_symbols: bool = typer.Option(
        False,
        "--all-symbols",
        help="Include all symbols in the log (default: only YAML symbol)",
    ),
) -> None:
    """Export experiments.jsonl to a wide CSV (flattened settings + metrics)."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    root = Path.cwd()
    log_path = experiments_log_path(cfg, base_dir=root)
    sym = None if all_symbols else cfg.symbol.upper()
    try:
        n = export_experiments_to_csv(
            log_path,
            output.resolve(),
            symbol_filter=sym,
        )
    except FileNotFoundError as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    typer.echo(str(output.resolve()))
    typer.echo(f"rows={n}")


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


@app.command()
def backtest(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (barriers, paths, day-trade cap)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run",
        help="Run id under artifacts/SYMBOL/ (default: latest with predictions)",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="Prediction split to backtest: val, train, or all",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        min=0.0,
        max=1.0,
        help="Enter when proba_take_profit >= threshold (overrides YAML default)",
    ),
    sweep: bool = typer.Option(
        False,
        "--sweep",
        help="Sweep thresholds; writes backtest_threshold_sweep.csv/json",
    ),
    sweep_step: float = typer.Option(
        0.05,
        "--sweep-step",
        min=0.01,
        max=0.5,
        help="Threshold grid step when --sweep (default 0.05)",
    ),
    sweep_min_signals: int = typer.Option(
        5,
        "--sweep-min-signals",
        min=1,
        help="Minimum signals for suggested threshold in sweep output",
    ),
    no_day_trade_cap: bool = typer.Option(
        False,
        "--no-day-trade-cap",
        help="Disable rolling day-trade cap when simulating entries",
    ),
) -> None:
    """Simulate policy PnL from predictions.parquet + labeled cache (Phase I1/I2)."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    root = Path.cwd()
    try:
        run_dir = resolve_run_dir(cfg, run_id, base_dir=root)
        if sweep:
            sweep_df, payload = run_threshold_sweep(
                cfg,
                run_dir,
                split=split,
                sweep_step=sweep_step,
                enforce_day_trade_cap=not no_day_trade_cap,
                min_signals_for_suggestion=sweep_min_signals,
                base_dir=root,
            )
        else:
            summary, _ = run_val_backtest(
                cfg,
                run_dir,
                split=split,
                tp_threshold=threshold,
                enforce_day_trade_cap=not no_day_trade_cap,
                base_dir=root,
            )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    if sweep:
        typer.echo(str((run_dir / "backtest_threshold_sweep.csv").resolve()))
        typer.echo(format_sweep_report(sweep_df, payload))
        return

    typer.echo(str((run_dir / "backtest_summary.json").resolve()))
    typer.echo(format_backtest_report(summary))


@meta_label_app.command("train")
def meta_label_train_cmd(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Experiment YAML (must match primary run features)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run",
        help="Primary run id with model_bundle.joblib",
    ),
    primary_threshold: float | None = typer.Option(
        None,
        "--primary-threshold",
        min=0.0,
        max=1.0,
        help="Primary proba_take_profit gate (default: YAML or 0.35)",
    ),
    meta_threshold: float | None = typer.Option(
        None,
        "--meta-threshold",
        min=0.0,
        max=1.0,
        help="Meta act probability floor used at compare time (default 0.5)",
    ),
) -> None:
    """Train binary meta-label model on primary-gated train rows."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    root = Path.cwd()
    try:
        run_dir = resolve_primary_run_dir(cfg, run_id, base_dir=root)
        bundle_path, metrics = train_meta_label(
            cfg,
            run_dir,
            primary_threshold=primary_threshold,
            meta_threshold=meta_threshold,
            base_dir=root,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    typer.echo(str(bundle_path.resolve()))
    typer.echo(
        f"meta_train_gated={metrics['n_meta_train_gated']}  "
        f"meta_train_positive={metrics['n_meta_train_positive']}  "
        f"primary_threshold={metrics['primary_threshold']}",
    )


@meta_label_app.command("compare")
def meta_label_compare_cmd(
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
        help="Primary run id (must have model + meta_label bundles)",
    ),
    primary_threshold: float | None = typer.Option(
        None,
        "--primary-threshold",
        min=0.0,
        max=1.0,
        help="Primary proba_take_profit gate",
    ),
    meta_threshold: float | None = typer.Option(
        None,
        "--meta-threshold",
        min=0.0,
        max=1.0,
        help="Meta act probability floor",
    ),
    no_day_trade_cap: bool = typer.Option(
        False,
        "--no-day-trade-cap",
        help="Disable day-trade cap in economics comparison",
    ),
) -> None:
    """Compare argmax vs threshold vs meta-filter policies on val."""
    cfg_path = _resolve_config(config)
    cfg = load_experiment_config(cfg_path)
    root = Path.cwd()
    try:
        run_dir = resolve_primary_run_dir(cfg, run_id, base_dir=root)
        results = compare_entry_policies(
            cfg,
            run_dir,
            primary_threshold=primary_threshold,
            meta_threshold=meta_threshold,
            enforce_day_trade_cap=not no_day_trade_cap,
            base_dir=root,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    typer.echo(str((run_dir / "meta_label_compare.json").resolve()))
    typer.echo(format_compare_report(results))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
