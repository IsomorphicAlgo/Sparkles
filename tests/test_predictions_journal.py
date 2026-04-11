"""Predictions export and journal compare."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from sparkles.config.schema import ExperimentConfig, JournalConfig, TrainConfig
from sparkles.journal.compare import (
    aggregate_predictions_by_session,
    load_and_normalize_journal,
    run_journal_compare,
)
from sparkles.models.predictions_export import predictions_frame
from sparkles.models.train import run_train


def _cfg(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "symbol": "T",
        "data_start": date(2024, 6, 1),
        "data_end": date(2024, 6, 10),
        "train_start": date(2024, 6, 3),
        "train_end": date(2024, 6, 4),
        "val_start": date(2024, 6, 5),
        "val_end": date(2024, 6, 7),
    }
    base.update(kw)
    return ExperimentConfig(**base)


def test_predictions_frame_has_proba_and_session_date() -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
        ],
    )
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0]}, index=idx)
    y_str = pd.Series(["take_profit", "stop_loss"], index=idx)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_e = le.fit_transform(y_str)
    est = LogisticRegression(max_iter=500, random_state=0)
    est.fit(X.values, y_e)
    pred = est.predict(X.values)
    df = predictions_frame(X, y_str, y_e, pred, "val", est, le, tz)
    assert len(df) == 2
    assert set(df["split"].unique()) == {"val"}
    assert "max_proba" in df.columns
    assert "session_date" in df.columns
    assert df["y_true"].tolist() == ["take_profit", "stop_loss"]


def test_run_train_writes_predictions_parquet_when_val(tmp_path: Path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg()
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    pred_p = out / "predictions.parquet"
    assert pred_p.is_file()
    pred = pd.read_parquet(pred_p)
    assert "entry_time" in pred.columns
    assert "y_true" in pred.columns and "y_pred" in pred.columns
    assert pred["split"].eq("val").all()


def test_run_train_skips_predictions_when_none(tmp_path: Path) -> None:
    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 10:00", tz=tz),
            pd.Timestamp("2024-06-03 11:00", tz=tz),
            pd.Timestamp("2024-06-04 10:00", tz=tz),
            pd.Timestamp("2024-06-04 11:00", tz=tz),
            pd.Timestamp("2024-06-05 10:00", tz=tz),
            pd.Timestamp("2024-06-05 11:00", tz=tz),
            pd.Timestamp("2024-06-06 10:00", tz=tz),
            pd.Timestamp("2024-06-06 11:00", tz=tz),
        ],
    )
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1e6] * 8,
        },
        index=idx,
    )
    outcomes = [
        "take_profit",
        "stop_loss",
        "vertical",
        "end_of_data",
        "take_profit",
        "stop_loss",
        "vertical",
        "take_profit",
    ]
    labels = pd.DataFrame(
        {
            "entry_close": [100.0] * 8,
            "barrier_outcome": outcomes,
            "sigma_ann_at_entry": [0.6] * 8,
            "vol_scale_ratio": [1.0] * 8,
            "tp_move_effective": [0.1] * 8,
            "sl_move": [0.05] * 8,
        },
        index=idx,
    )
    cfg = _cfg(train=TrainConfig(export_predictions="none"))
    out = run_train(cfg, base_dir=tmp_path, labels=labels, ohlcv=ohlcv)
    assert not (out / "predictions.parquet").is_file()


def test_journal_compare_merges_by_entry_date(tmp_path: Path) -> None:
    cfg = _cfg(
        symbol="ZZ",
        journal=JournalConfig(csv_path="trades.csv"),
    )
    run_dir = tmp_path / "artifacts" / "ZZ" / "run1"
    run_dir.mkdir(parents=True)
    tz = "America/New_York"
    pred = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2024-06-05 10:00:00"],
            ).tz_localize(tz),
            "session_date": [date(2024, 6, 5)],
            "split": ["val"],
            "y_true": ["take_profit"],
            "y_pred": ["take_profit"],
            "max_proba": [0.9],
        },
    )
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    jpath = tmp_path / "trades.csv"
    jpath.write_text("entry_date,symbol\n2024-06-05,ZZ\n", encoding="utf-8")

    merged, out_csv = run_journal_compare(
        cfg,
        run_dir,
        split_filter="val",
        base_dir=tmp_path,
    )
    assert out_csv.is_file()
    assert len(merged) == 1
    assert bool(merged["model_matched"].iloc[0])
    assert merged["pred_n"].iloc[0] == 1


def test_aggregate_predictions_by_session_mode() -> None:
    pred = pd.DataFrame(
        {
            "entry_time": [1, 2, 3],
            "session_date": [date(2024, 1, 1)] * 3,
            "split": ["val"] * 3,
            "y_true": ["a", "a", "a"],
            "y_pred": ["a", "b", "b"],
            "max_proba": [0.5, 0.6, 0.7],
        },
    )
    g = aggregate_predictions_by_session(pred, split_filter="val")
    assert len(g) == 1
    assert g["pred_n"].iloc[0] == 3
    assert g["y_pred_mode"].iloc[0] == "b"


def test_load_and_normalize_journal_filters_symbol(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text(
        "entry_date,symbol\n2024-01-02,ZZ\n2024-01-03,OTHER\n",
        encoding="utf-8",
    )
    j = load_and_normalize_journal(p, "ZZ")
    assert len(j) == 1
    assert j["entry_date"].iloc[0] == date(2024, 1, 2)
