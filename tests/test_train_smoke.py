"""End-to-end train with injected frames (no large Parquet fixtures)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sparkles.config.schema import ExperimentConfig
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


def test_run_train_writes_artifacts(tmp_path) -> None:
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
    assert out.is_dir()
    assert (out / "model_bundle.joblib").is_file()
    assert (out / "metrics.json").is_file()
    log = tmp_path / "artifacts" / "experiments.jsonl"
    assert log.is_file()
