"""Phase 1 artifact / cache summary for ``sparkles report`` (Iteration 7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparkles.config.schema import ExperimentConfig
from sparkles.data.ingest import parquet_cache_path
from sparkles.labels.triple_barrier import labeled_parquet_path


def _json_compact(obj: Any) -> str:
    """Single-line JSON for terminal (dates and other types via default=str)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _format_yaml_parameters(cfg: ExperimentConfig) -> list[str]:
    """Human-readable model / train / feature lines from the loaded experiment YAML."""
    out: list[str] = []
    out.append("parameters (current experiment YAML):")
    if (
        cfg.train_start is not None
        and cfg.train_end is not None
        and cfg.val_start is not None
        and cfg.val_end is not None
    ):
        out.append(
            f"  splits: train={cfg.train_start}..{cfg.train_end}  "
            f"val={cfg.val_start}..{cfg.val_end}",
        )
    else:
        out.append("  splits: (train_start/train_end/val_start/val_end not all set)")

    out.append(
        "  labeling: "
        f"profit_barrier_base={cfg.profit_barrier_base}  "
        f"stop_loss_base={cfg.stop_loss_base}  "
        f"min_profit_per_trade_pct={cfg.min_profit_per_trade_pct}  "
        f"label_entry_stride={cfg.label_entry_stride}  "
        f"vertical_max_trading_days={cfg.vertical_max_trading_days}  "
        f"vol_lookback_trading_days={cfg.vol_lookback_trading_days}",
    )
    out.append(
        "  ingest: "
        f"ingest_chunk_calendar_days={cfg.ingest_chunk_calendar_days}  "
        f"ingest_sleep_seconds_between_chunks={cfg.ingest_sleep_seconds_between_chunks}",
    )

    li = cfg.live_ingest
    out.append(
        "  live_ingest (Phase 2): "
        f"enabled={li.enabled}  poll_interval_seconds={li.poll_interval_seconds}  "
        f"refresh_lookback_calendar_days={li.refresh_lookback_calendar_days}  "
        f"merge_strategy={li.merge_strategy}  "
        f"include_extended_hours={li.include_extended_hours}  "
        f"session={li.session_start_local!r}..{li.session_end_local!r}",
    )

    mc = cfg.model
    out.append(
        "  model: "
        f"type={mc.type}  solver={mc.solver}  tol={mc.tol}  "
        f"logistic_c={mc.logistic_c}  max_iter={mc.max_iter}  "
        f"random_seed={mc.random_seed}  class_weight={mc.class_weight!r}",
    )

    tc = cfg.train
    out.append(
        "  train: "
        f"min_train_rows={tc.min_train_rows}  min_val_rows={tc.min_val_rows}  "
        f"drop_val_unseen_classes={tc.drop_val_unseen_classes}  "
        f"experiment_name={tc.experiment_name!r}  notes={tc.notes!r}",
    )
    out.append(f"  features: {_json_compact(cfg.features.model_dump())}")
    return out


def _format_metrics_block(m: dict[str, Any]) -> list[str]:
    """Metrics.json headline + classes + stored feature flags (if present)."""
    lines: list[str] = []
    mt = m.get("model_type")
    if mt is not None:
        lines.append(f"  model_type (stored): {mt}")
    else:
        lines.append("  model_type (stored): (absent)")
    lines.append(
        "  metrics: "
        f"train_acc={m.get('train_accuracy')}  val_acc={m.get('val_accuracy')}  "
        f"train_n={m.get('train_n')}  val_n={m.get('val_n')}",
    )
    classes = m.get("classes")
    if isinstance(classes, list) and classes:
        lines.append(f"  classes: {_json_compact(classes)}")
    feats = m.get("features")
    if isinstance(feats, dict):
        lines.append(f"  features (stored in metrics): {_json_compact(feats)}")
    else:
        lines.append(
            "  features (stored in metrics): (absent — train predates feature logging)",
        )
    return lines


def _format_experiment_row(row: dict[str, Any]) -> str:
    """One compact line for experiments.jsonl tail."""
    rid = row.get("run_id", "?")
    va = row.get("val_accuracy", "?")
    mt = row.get("model_type", "?")
    ms = row.get("model_solver", "?")
    cw = row.get("model_class_weight", row.get("class_weight"))
    name = row.get("train_experiment_name")
    notes = row.get("train_notes")
    feats = row.get("features")
    parts = [
        f"  {rid}",
        f"val_acc={va}",
        f"model={mt}/{ms}",
        f"class_weight={cw!r}",
    ]
    if isinstance(feats, dict):
        parts.append(f"features={_json_compact(feats)}")
    if name is not None:
        parts.append(f"experiment_name={name!r}")
    if notes is not None:
        parts.append(f"notes={notes!r}")
    return "  ".join(parts)


def _runs_with_metrics(symbol_dir: Path) -> list[Path]:
    if not symbol_dir.is_dir():
        return []
    out: list[Path] = []
    for p in symbol_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").is_file():
            out.append(p)
    return sorted(out, key=lambda x: x.name, reverse=True)


def _tail_experiments_for_symbol(
    log_path: Path,
    symbol: str,
    *,
    max_entries: int = 5,
) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    symu = symbol.upper()
    raw = log_path.read_text(encoding="utf-8").splitlines()
    picked: list[dict[str, Any]] = []
    for line in reversed(raw[-500:]):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(obj.get("symbol", "")).upper() == symu:
            picked.append(obj)
            if len(picked) >= max_entries:
                break
    return list(reversed(picked))


def run_phase1_report(
    cfg: ExperimentConfig,
    *,
    base_dir: Path | None = None,
    run_id: str | None = None,
) -> str:
    """Build a multi-line human-readable summary (no trailing newline required)."""
    root = Path.cwd() if base_dir is None else base_dir
    lines: list[str] = []

    lines.append(
        f"experiment: symbol={cfg.symbol.upper()}  "
        f"data_range={cfg.data_start}..{cfg.data_end}",
    )
    lines.extend(_format_yaml_parameters(cfg))

    ingest_p = parquet_cache_path(cfg, base_dir=base_dir)
    label_p = labeled_parquet_path(cfg, base_dir=base_dir)
    ir = ingest_p.resolve()
    lr = label_p.resolve()
    lines.append(f"ingest_cache: exists={ingest_p.is_file()}  path={ir}")
    lines.append(f"labeled:     exists={label_p.is_file()}  path={lr}")

    art_sym = root / cfg.paths.artifacts_dir / cfg.symbol.upper()
    metrics_path: Path | None = None
    if run_id:
        cand = art_sym / run_id
        if (cand / "metrics.json").is_file():
            metrics_path = cand / "metrics.json"
            lines.append(f"train_run:   (explicit) {cand.resolve()}")
        else:
            lines.append(
                f"train_run:   --run {run_id!r} not found or missing metrics.json",
            )
    else:
        runs = _runs_with_metrics(art_sym)
        if runs:
            metrics_path = runs[0] / "metrics.json"
            lines.append(f"train_run:   latest {runs[0].resolve()}")
        else:
            lines.append(
                "train_run:   (none — run sparkles train after ingest and label)",
            )

    if metrics_path is not None:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.extend(_format_metrics_block(m))

    exp_log = root / cfg.paths.artifacts_dir / "experiments.jsonl"
    tail = _tail_experiments_for_symbol(exp_log, cfg.symbol)
    if tail:
        er = exp_log.resolve()
        lines.append(f"experiments_log: {er} (last {len(tail)} for symbol)")
        for row in tail:
            lines.append(_format_experiment_row(row))
    elif exp_log.is_file():
        er = exp_log.resolve()
        lines.append(f"experiments_log: {er} (no rows for this symbol yet)")
    else:
        lines.append("experiments_log: (file not created yet)")

    return "\n".join(lines)
