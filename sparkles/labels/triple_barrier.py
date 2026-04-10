"""Triple-barrier labels with vol scaling and min-profit floor (Iteration 4).

Entry: ``close`` at sampled bar *i*. Barriers scale with ``sigma_t / sigma_ref``
(clamped), take-profit move floored by ``min_profit_per_trade_pct``. Forward scan
uses 1m ``high`` / ``low``; same bar: **stop** before **take-profit**
(pessimistic long).

If you change labeling horizons or barrier math, update configs/experiments/*.yaml and
``sparkles/features/volatility.py`` (sigma alignment).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sparkles.features.volatility import ensure_exchange_tz_index
from sparkles.labels.types import BarrierOutcome

if TYPE_CHECKING:
    from sparkles.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)

_OUTCOME_TO_CODE = {
    BarrierOutcome.TAKE_PROFIT: np.int8(0),
    BarrierOutcome.STOP_LOSS: np.int8(1),
    BarrierOutcome.VERTICAL: np.int8(2),
    BarrierOutcome.END_OF_DATA: np.int8(3),
}
_CODE_TO_OUTCOME = {v: k for k, v in _OUTCOME_TO_CODE.items()}


def _ann_vol_column_name(lookback: int) -> str:
    return f"vol_{lookback}d_ann"


def _sigma_ref(vol_ann: NDArray[np.float64], method: str) -> float:
    v = vol_ann[np.isfinite(vol_ann)]
    if v.size == 0:
        raise ValueError("No finite vol_lookback values to form sigma_ref")
    if method == "median":
        return float(np.median(v))
    if method == "mean":
        return float(np.mean(v))
    raise ValueError(f"Unknown vol_ref_method: {method}")


def _trading_day_ranks(
    index: pd.DatetimeIndex | pd.Index,
    exchange_timezone: str,
) -> NDArray[np.int32]:
    """Integer rank 0..K-1 per bar by ascending session date in exchange TZ."""
    ix = ensure_exchange_tz_index(index, exchange_timezone)
    norm = pd.DatetimeIndex(ix).normalize()
    uniq = norm.unique().sort_values()
    mapper = {ts: np.int32(i) for i, ts in enumerate(uniq)}
    return np.array([mapper[t] for t in norm], dtype=np.int32)


def _scan_single_entry(
    i: int,
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    day_rank: NDArray[np.int32],
    upper: float,
    lower: float,
    entry_day_rank: int,
    vertical_max_trading_days: int,
    n: int,
) -> tuple[BarrierOutcome, int]:
    """Return (outcome, bars_forward) where bars_forward counts bars after entry."""
    for step, j in enumerate(range(i + 1, n), start=1):
        if day_rank[j] - entry_day_rank >= vertical_max_trading_days:
            return BarrierOutcome.VERTICAL, step
        if low[j] <= lower:
            return BarrierOutcome.STOP_LOSS, step
        if high[j] >= upper:
            return BarrierOutcome.TAKE_PROFIT, step
    return BarrierOutcome.END_OF_DATA, n - i - 1


def build_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    ann_col: str | None = None,
) -> pd.DataFrame:
    """One row per strided entry with outcome, levels, and diagnostics.

    Expects columns ``open``, ``high``, ``low``, ``close`` and annualized vol
    ``vol_{N}d_ann`` (add via ``add_volatility_from_config`` if missing).
    """
    need = {"high", "low", "close"}
    missing = need - set(ohlcv.columns)
    if missing:
        raise KeyError(f"OHLCV missing columns: {sorted(missing)}")

    lookback = cfg.vol_lookback_trading_days
    if ann_col is None:
        ann_col = _ann_vol_column_name(lookback)
    if ann_col not in ohlcv.columns:
        raise KeyError(
            f"Missing {ann_col!r}; run add_volatility_from_config(ohlcv, cfg) first",
        )

    high = ohlcv["high"].to_numpy(dtype=np.float64, copy=False)
    low = ohlcv["low"].to_numpy(dtype=np.float64, copy=False)
    close = ohlcv["close"].to_numpy(dtype=np.float64, copy=False)
    vol_ann = ohlcv[ann_col].to_numpy(dtype=np.float64, copy=False)
    n = len(ohlcv)
    if n < 3:
        raise ValueError("Need at least 3 rows to label")

    sigma_ref = _sigma_ref(vol_ann, cfg.vol_ref_method)
    day_rank = _trading_day_ranks(ohlcv.index, cfg.exchange_timezone)

    stride = cfg.label_entry_stride
    entry_positions = np.arange(0, n - 1, stride, dtype=np.int64)
    logger.info(
        "Labeling %s entries (stride=%s, n_bars=%s)",
        len(entry_positions),
        stride,
        n,
    )

    rows: list[dict[str, object]] = []
    lo = cfg.barrier_vol_scale_min
    hi = cfg.barrier_vol_scale_max
    for k, i in enumerate(entry_positions):
        if (k + 1) % 10000 == 0:
            logger.info("Labeled %s / %s entries", k + 1, len(entry_positions))
        sig_t = vol_ann[i]
        if not np.isfinite(sig_t) or not np.isfinite(close[i]):
            continue
        ratio_raw = sig_t / sigma_ref
        ratio = float(np.clip(ratio_raw, lo, hi))
        tp_move = cfg.profit_barrier_base * ratio
        sl_move = cfg.stop_loss_base * ratio
        eff_tp = max(cfg.min_profit_per_trade_pct, tp_move)
        entry_px = float(close[i])
        upper = entry_px * (1.0 + eff_tp)
        lower = entry_px * (1.0 - sl_move)
        entry_rank = int(day_rank[i])
        out, bars_f = _scan_single_entry(
            i,
            high,
            low,
            close,
            day_rank,
            upper,
            lower,
            entry_rank,
            cfg.vertical_max_trading_days,
            n,
        )
        rows.append(
            {
                "entry_time": ohlcv.index[i],
                "entry_close": entry_px,
                "barrier_outcome": out.value,
                "bars_forward": int(bars_f),
                "sigma_ann_at_entry": float(sig_t),
                "sigma_ref": float(sigma_ref),
                "vol_scale_ratio": ratio,
                "tp_move_effective": float(eff_tp),
                "sl_move": float(sl_move),
                "upper_barrier": float(upper),
                "lower_barrier": float(lower),
            },
        )

    return cast(pd.DataFrame, pd.DataFrame(rows).set_index("entry_time"))


def labeled_parquet_path(cfg: ExperimentConfig, base_dir: Path | None = None) -> Path:
    root = Path.cwd() if base_dir is None else base_dir
    cache = root / cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    name = (
        f"{cfg.symbol.upper()}_labeled_{cfg.data_start.isoformat()}_"
        f"{cfg.data_end.isoformat()}_s{cfg.label_entry_stride}.parquet"
    )
    return cache / name


def run_label(
    cfg: ExperimentConfig,
    *,
    ohlcv: pd.DataFrame | None = None,
    parquet_path: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Load OHLCV (or Parquet), ensure vol columns, write labeled Parquet."""
    from sparkles.data.ingest import parquet_cache_path
    from sparkles.features import add_volatility_from_config

    if ohlcv is None:
        pq = parquet_path or parquet_cache_path(cfg, base_dir=base_dir)
        if not pq.is_file():
            raise FileNotFoundError(f"Ingest Parquet not found: {pq}")
        ohlcv = pd.read_parquet(pq)

    ann_col = _ann_vol_column_name(cfg.vol_lookback_trading_days)
    if ann_col not in ohlcv.columns:
        ohlcv = add_volatility_from_config(ohlcv, cfg)

    labels = build_triple_barrier_labels(ohlcv, cfg, ann_col=ann_col)
    out = labeled_parquet_path(cfg, base_dir=base_dir)
    labels.to_parquet(out, engine="pyarrow")
    logger.info("Wrote %s label rows to %s", len(labels), out)
    return out
