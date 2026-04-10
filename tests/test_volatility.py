"""Volatility: no lookahead and alignment to 1m bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparkles.features.volatility import (
    TRADING_DAYS_PER_YEAR,
    add_volatility_columns,
    align_volatility_to_1m_index,
    daily_last_close,
    daily_log_returns,
    ensure_exchange_tz_index,
    rolling_volatility_daily_returns_no_lookahead,
)


def test_ensure_tz_localizes_naive_as_exchange() -> None:
    idx = pd.DatetimeIndex(["2024-06-01 09:30:00", "2024-06-01 09:31:00"])
    out = ensure_exchange_tz_index(idx, "America/New_York")
    assert str(out.tz) == "America/New_York"


def test_daily_last_close_takes_last_bar_of_session() -> None:
    tz = "America/New_York"
    t0 = pd.Timestamp("2024-06-03 09:30", tz=tz)
    idx = [t0 + pd.Timedelta(minutes=i) for i in range(5)]
    df = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 99.0]}, index=idx)
    d = daily_last_close(df, exchange_timezone=tz)
    assert len(d) == 1
    assert float(d.iloc[0]) == 99.0


def test_rolling_vol_shift_excludes_same_day_close() -> None:
    """Vol on session D uses returns through D-1 only (shift(1) after rolling)."""
    tz = "America/New_York"
    days = pd.bdate_range("2024-01-02", periods=25, tz=tz)
    # Strong positive drift then massive jump on last day close
    values = [100.0 + float(i) for i in range(24)] + [500.0]
    daily_close = pd.Series(values, index=days)
    sig_d, sig_a = rolling_volatility_daily_returns_no_lookahead(
        daily_close,
        lookback_trading_days=5,
        annualize=True,
    )
    # Last day: shifted vol excludes the final huge return; unshifted rolling does not.
    last_sig = float(sig_a.iloc[-1])
    # "Leaked" vol at last close: rolling std without shift(1).
    r = daily_log_returns(daily_close)
    leaked = r.rolling(5, min_periods=5).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert not np.isnan(last_sig)
    assert not np.isnan(leaked)
    assert last_sig != pytest.approx(float(leaked), rel=1e-6, abs=1e-9)


def test_intraday_bars_on_spike_day_share_pre_spike_vol() -> None:
    """All 1m bars on the spike session share vol that ignores that session's close."""
    tz = "America/New_York"
    sessions = list(pd.bdate_range("2024-01-02", periods=22, tz=tz))
    rows: list[tuple[pd.Timestamp, float]] = []
    for i, day in enumerate(sessions):
        open_t = day.replace(hour=9, minute=30)
        # Two bars per day: same close except last session last bar spikes
        c1, c2 = (100.0 + float(i), 100.0 + float(i))
        if i == len(sessions) - 1:
            c2 = 500.0
        rows.append((open_t, c1))
        rows.append((open_t + pd.Timedelta(minutes=1), c2))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    df = pd.DataFrame([r[1] for r in rows], index=idx, columns=["close"])
    out = add_volatility_columns(
        df,
        exchange_timezone=tz,
        lookback_trading_days=5,
        annualize=True,
    )
    spike_day = sessions[-1].normalize()
    mask = ensure_exchange_tz_index(out.index, tz).normalize() == spike_day
    vols = out.loc[mask, "vol_5d_ann"].astype(float)
    assert vols.nunique() == 1
    assert vols.notna().all()


def test_align_volatility_broadcast_matches_length() -> None:
    tz = "America/New_York"
    day = pd.Timestamp("2024-01-02 09:30", tz=tz)
    idx = day + pd.to_timedelta(np.arange(10), "m")
    ohlcv = pd.DataFrame({"close": 100.0}, index=idx)
    sigma = pd.Series([0.42], index=[day.normalize()])
    aligned = align_volatility_to_1m_index(
        ohlcv,
        sigma,
        exchange_timezone=tz,
    )
    assert len(aligned) == len(ohlcv)
    assert (aligned == 0.42).all()
