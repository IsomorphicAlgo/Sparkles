"""Parameter grid helpers for batch training (ML expansion Phase E)."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from sparkles.config.load import load_experiment_config, load_experiment_config_merged
from sparkles.config.schema import ExperimentConfig


def set_by_dot_path(root: dict[str, Any], path: str, value: Any) -> None:
    """Set ``value`` on a nested dict using dotted keys (e.g. ``model.xgb_max_depth``)."""
    parts = path.split(".")
    if not parts or not parts[0]:
        raise ValueError(f"Invalid dot path: {path!r}")
    cur: dict[str, Any] = root
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(
                f"Cannot set {path!r}: {'.'.join(parts[: parts.index(part) + 1])!r} "
                f"is not a mapping",
            )
        cur = nxt
    cur[parts[-1]] = value


def apply_dot_path_overrides(
    base: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deep copy of ``base`` with dotted-path overrides applied."""
    out = copy.deepcopy(base)
    for path, value in overrides.items():
        set_by_dot_path(out, str(path), value)
    return out


def expand_param_grid(params: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``{dot_path: [values, ...]}`` → list of override dicts."""
    if not params:
        return [{}]
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    if any(not isinstance(v, list) for v in value_lists):
        bad = [k for k, v in zip(keys, value_lists) if not isinstance(v, list)]
        raise ValueError(f"Grid param values must be lists; got non-list for: {bad}")
    combos: list[dict[str, Any]] = []
    for values in product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos


def grid_experiment_suffix(overrides: Mapping[str, Any]) -> str:
    """Compact suffix from grid overrides for ``train.experiment_name``."""
    parts: list[str] = []
    for path, value in sorted(overrides.items()):
        leaf = str(path).split(".")[-1]
        if isinstance(value, float):
            token = f"{value:g}".replace(".", "p")
        elif isinstance(value, bool):
            token = "1" if value else "0"
        else:
            token = str(value).replace(".", "p").replace(" ", "")
        parts.append(f"{leaf}{token}")
    return "_".join(parts)


def load_grid_spec(path: Path | str) -> dict[str, Any]:
    """Load a grid-search YAML spec (see ``configs/experiments/grids/``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Grid spec not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None or not isinstance(raw, dict):
        raise ValueError(f"Grid spec root must be a mapping: {p}")
    return raw


def resolve_grid_base_dict(
    *,
    base_path: Path | str,
    preset_path: Path | str | None = None,
) -> dict[str, Any]:
    """Merge base (+ optional preset) experiment YAML into a plain dict."""
    if preset_path is None:
        cfg = load_experiment_config(base_path)
    else:
        cfg = load_experiment_config_merged(base_path, preset_path)
    return cfg.model_dump(mode="json")


def build_grid_configs(
    spec: Mapping[str, Any],
    *,
    base_path: Path | str,
    preset_path: Path | str | None = None,
) -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """Expand ``spec['params']`` and return (override dict, validated config) pairs."""
    preset = preset_path or spec.get("preset")
    base_dict = resolve_grid_base_dict(
        base_path=base_path,
        preset_path=Path(preset) if preset else None,
    )
    fixed = dict(spec.get("fixed") or {})
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("Grid spec 'params' must be a mapping of dot_path -> [values]")
    if not isinstance(fixed, dict):
        raise ValueError("Grid spec 'fixed' must be a mapping of dot_path -> value")

    prefix = str(spec.get("experiment_name_prefix") or spec.get("grid_name") or "grid")
    notes_prefix = spec.get("notes_prefix")

    out: list[tuple[dict[str, Any], ExperimentConfig]] = []
    for combo in expand_param_grid(params):
        merged = apply_dot_path_overrides(base_dict, fixed)
        merged = apply_dot_path_overrides(merged, combo)
        suffix = grid_experiment_suffix(combo)
        merged.setdefault("train", {})
        if not isinstance(merged["train"], dict):
            raise ValueError("train section must be a mapping after overrides")
        merged["train"]["experiment_name"] = f"{prefix}_{suffix}"
        if notes_prefix:
            merged["train"]["notes"] = f"{notes_prefix} | {suffix}"
        cfg = ExperimentConfig.model_validate(merged)
        out.append((combo, cfg))
    return out
