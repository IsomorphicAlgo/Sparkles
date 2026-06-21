"""Day-trade experiment YAML loads and uses distinct label cache path."""

from __future__ import annotations

from pathlib import Path

from sparkles.config.load import load_experiment_config
from sparkles.labels.triple_barrier import labeled_parquet_path


def test_rklb_daytrade_v1_config_loads() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "experiments" / "rklb_daytrade_v1.yaml"
    cfg = load_experiment_config(path)
    assert cfg.symbol == "RKLB"
    assert cfg.profit_barrier_base == 0.15
    assert cfg.stop_loss_base == 0.10
    assert cfg.min_profit_per_trade_pct == 0.12
    assert cfg.vertical_max_trading_days == 1
    assert cfg.label_entry_stride == 15
    assert cfg.train.experiment_name == "rklb_daytrade_v1"


def test_rklb_daytrade_v1_labeled_path_stride() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(repo / "configs" / "experiments" / "rklb_daytrade_v1.yaml")
    out = labeled_parquet_path(cfg, base_dir=repo)
    assert out.name.endswith("_s15.parquet")
    assert "RKLB_labeled" in out.name


def test_rklb_daytrade_v2_config_and_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "experiments" / "rklb_daytrade_v2.yaml"
    cfg = load_experiment_config(path)
    assert cfg.profit_barrier_base == 0.03
    assert cfg.stop_loss_base == 0.05
    assert cfg.min_profit_per_trade_pct == 0.05
    assert cfg.vertical_max_trading_days == 1
    assert cfg.label_entry_stride == 15
    assert cfg.label_cache_suffix == "dt_v2"
    assert cfg.train.experiment_name == "rklb_daytrade_v2"
    out = labeled_parquet_path(cfg, base_dir=repo)
    assert out.name == "RKLB_labeled_2022-01-01_2026-03-30_s15_dt_v2.parquet"
