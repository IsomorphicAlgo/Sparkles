"""Technical indicators at entry time (ML expansion Phase G4a).

EMA distance, Wilder RSI, and MACD computed on the full 1m close series; values
are sliced at each labeled entry bar (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparkles.config.schema import FeatureConfig
from sparkles.features.builders import EntryFeatureContext
from sparkles.features.intraday import _slice_at_entries


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(np.float64).ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.astype(np.float64).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_technical_indicators(ctx: EntryFeatureContext) -> pd.DataFrame:
    fc = ctx.feature_config
    close = ctx.full_ohlcv["close"]
    pos = ctx.entry_positions
    cols: dict[str, pd.Series] = {}

    for w in fc.ema_windows_bars:
        ema = _ema(close, w)
        dist = (close.astype(np.float64) - ema) / ema.clip(lower=1e-12)
        cols[f"ema_dist_{w}m"] = _slice_at_entries(dist, pos)

    rsi = _wilder_rsi(close, fc.rsi_window_bars)
    cols[f"rsi_{fc.rsi_window_bars}m"] = _slice_at_entries(rsi / 100.0, pos)

    ema_fast = _ema(close, fc.macd_fast_bars)
    ema_slow = _ema(close, fc.macd_slow_bars)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(
        span=fc.macd_signal_bars,
        adjust=False,
        min_periods=fc.macd_signal_bars,
    ).mean()
    macd_hist = macd_line - macd_signal
    cols["macd_line"] = _slice_at_entries(macd_line, pos)
    cols["macd_signal"] = _slice_at_entries(macd_signal, pos)
    cols["macd_hist"] = _slice_at_entries(macd_hist, pos)

    return pd.DataFrame(cols, index=pos.index)


def g4a_warmup_bars(fc: FeatureConfig) -> int:
    """Minimum full-OHLCV bars before G4a indicators are defined at entry."""
    if not fc.technical_indicators:
        return 0
    need = max(fc.ema_windows_bars) if fc.ema_windows_bars else 0
    need = max(need, fc.rsi_window_bars)
    need = max(need, fc.macd_slow_bars + fc.macd_signal_bars)
    return need
