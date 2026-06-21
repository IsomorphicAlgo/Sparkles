"""Parameter grid helpers for batch training."""

from __future__ import annotations

from sparkles.config.grid import (
    apply_dot_path_overrides,
    build_grid_configs,
    expand_param_grid,
    grid_experiment_suffix,
    set_by_dot_path,
)


def test_set_by_dot_path_nested() -> None:
    root: dict = {"model": {"type": "x"}}
    set_by_dot_path(root, "model.xgb_max_depth", 4)
    assert root["model"]["xgb_max_depth"] == 4


def test_expand_param_grid_cartesian() -> None:
    combos = expand_param_grid(
        {
            "model.xgb_max_depth": [3, 4],
            "model.xgb_learning_rate": [0.08, 0.02],
        },
    )
    assert len(combos) == 4
    assert {"model.xgb_max_depth": 3, "model.xgb_learning_rate": 0.08} in combos


def test_apply_dot_path_overrides_copies_base() -> None:
    base = {"model": {"type": "logistic_regression", "tol": 1e-4}, "symbol": "RKLB"}
    out = apply_dot_path_overrides(base, {"model.class_weight": "balanced"})
    assert base["model"] == {"type": "logistic_regression", "tol": 1e-4}
    assert out["model"]["class_weight"] == "balanced"


def test_grid_experiment_suffix() -> None:
    s = grid_experiment_suffix(
        {"model.xgb_max_depth": 3, "model.xgb_learning_rate": 0.08},
    )
    assert "xgb_max_depth3" in s
    assert "xgb_learning_rate0p08" in s


def test_build_grid_configs(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
symbol: RKLB
data_start: 2024-01-01
data_end: 2024-06-01
model:
  type: logistic_regression
features:
  log_entry_close: true
""",
        encoding="utf-8",
    )
    spec = {
        "experiment_name_prefix": "testgrid",
        "params": {"model.class_weight": ["balanced", None]},
        "fixed": {"train.export_predictions": "none"},
    }
    pairs = build_grid_configs(spec, base_path=base)
    assert len(pairs) == 2
    names = {cfg.train.experiment_name for _, cfg in pairs}
    assert all(n.startswith("testgrid_") for n in names)
    assert all(cfg.train.export_predictions == "none" for _, cfg in pairs)
