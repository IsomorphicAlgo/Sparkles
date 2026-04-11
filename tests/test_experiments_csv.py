"""experiments.jsonl → CSV export and flatten helpers."""

from __future__ import annotations

import json

import pandas as pd

from sparkles.tracking.experiments_csv import (
    export_experiments_to_csv,
    flatten_log_row,
)


def test_flatten_log_row_nested() -> None:
    row = {
        "run_id": "x",
        "experiment_config": {"model": {"type": "logistic_regression", "tol": 1e-4}},
        "features": {"a": True},
    }
    flat = flatten_log_row(row)
    assert flat["run_id"] == "x"
    assert flat["experiment_config.model.type"] == "logistic_regression"
    assert flat["experiment_config.model.tol"] == 1e-4
    assert flat["features.a"] is True


def test_flatten_log_row_list_to_json_string() -> None:
    flat = flatten_log_row({"classes": ["a", "b"]})
    assert json.loads(flat["classes"]) == ["a", "b"]


def test_export_experiments_to_csv_filter_and_merge_columns(tmp_path) -> None:
    log = tmp_path / "experiments.jsonl"
    log.write_text(
        json.dumps({"symbol": "RKLB", "val_accuracy": 0.5, "run_id": "1"})
        + "\n"
        + json.dumps({"symbol": "OTHER", "val_accuracy": 0.9, "run_id": "2"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    n = export_experiments_to_csv(log, out, symbol_filter="RKLB")
    assert n == 1
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "RKLB"
    assert df["val_accuracy"].iloc[0] == 0.5


def test_export_empty_log_writes_csv(tmp_path) -> None:
    log = tmp_path / "experiments.jsonl"
    log.write_text("", encoding="utf-8")
    out = tmp_path / "empty.csv"
    n = export_experiments_to_csv(log, out, symbol_filter=None)
    assert n == 0
    assert out.is_file()
