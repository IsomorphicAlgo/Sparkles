"""Phase 1 report summary."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from sparkles.config.schema import ExperimentConfig
from sparkles.reporting.summary import run_phase1_report


def test_run_phase1_report_shows_latest_metrics(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        symbol="ZZ",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 30),
    )
    sym = tmp_path / "artifacts" / "ZZ"
    sym.mkdir(parents=True)
    run_dir = sym / "20990101T000000_000000Z"
    run_dir.mkdir(parents=True)
    metrics = {
        "train_accuracy": 0.5,
        "val_accuracy": 0.4,
        "train_n": 10,
        "val_n": 5,
        "classes": ["a"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    text = run_phase1_report(cfg, base_dir=tmp_path)
    assert "20990101T000000_000000Z" in text
    assert "val_acc=0.4" in text or "val_acc=0.4 " in text
    assert "train_n=10" in text


def test_run_phase1_report_explicit_run_id(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        symbol="ZZ",
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 30),
    )
    sym = tmp_path / "artifacts" / "ZZ"
    sym.mkdir(parents=True)
    (sym / "run_a").mkdir()
    (sym / "run_b").mkdir()
    m = {"train_accuracy": 1.0, "val_accuracy": 0.9, "train_n": 1, "val_n": 1}
    (sym / "run_b" / "metrics.json").write_text(
        json.dumps(m),
        encoding="utf-8",
    )
    text = run_phase1_report(cfg, base_dir=tmp_path, run_id="run_b")
    assert "run_b" in text
    assert "val_acc=0.9" in text
