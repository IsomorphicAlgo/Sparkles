"""Timezone alignment for market_context SPY join."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from sparkles.config.schema import ExperimentConfig
from sparkles.features.dataset import build_feature_matrix
from sparkles.features.market_data import load_market_context_frames
from sparkles.features.volatility import ensure_exchange_tz_index
from tests.test_g3_features import _synthetic_ohlcv, _write_context_parquets
from tests.test_dataset import _cfg


def test_spy_alignment_with_naive_label_index(tmp_path) -> None:
    """Labeled Parquet is tz-naive; SPY cache is tz-aware — must still join."""
    ohlcv = _synthetic_ohlcv()
    ohlcv_naive = ohlcv.copy()
    ohlcv_naive.index = ohlcv_naive.index.tz_localize(None)
    _write_context_parquets(tmp_path, ohlcv)
    labels = pd.DataFrame(
        {
            "entry_close": ohlcv_naive["close"].iloc[80:],
            "barrier_outcome": ["vertical"] * 40,
            "sigma_ann_at_entry": [0.5] * 40,
            "vol_scale_ratio": [1.0] * 40,
            "tp_move_effective": [0.1] * 40,
            "sl_move": [0.05] * 40,
        },
        index=ohlcv_naive.index[80:],
    )
    cfg = _cfg(
        data_start=date(2024, 6, 1),
        data_end=date(2024, 6, 10),
        context_ingest={
            "symbols": [
                {"symbol": "SPY", "interval": "1min"},
                {"symbol": "VIXY", "interval": "1day", "twelvedata_exchange": "CBOE"},
            ],
        },
        features={
            "log_entry_close": False,
            "label_geometry": False,
            "intraday_range_pct": False,
            "log1p_volume": False,
            "market_context": True,
            "market_spy_return_bars": 15,
        },
    )
    spy, _vix = load_market_context_frames(cfg, base_dir=tmp_path)
    entry_ix = ensure_exchange_tz_index(labels.index, cfg.exchange_timezone)
    spy_ix = ensure_exchange_tz_index(spy.index, cfg.exchange_timezone)
    matched = int((spy_ix.get_indexer(entry_ix) >= 0).sum())
    assert matched > 0

    X, _y = build_feature_matrix(labels, ohlcv_naive, cfg, base_dir=tmp_path)
    assert len(X) > 0
    assert "spy_ret_15m" in X.columns
    assert not X["spy_ret_15m"].isna().all()
