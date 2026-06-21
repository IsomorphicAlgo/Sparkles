"""TwelveData symbol guidance (avoids ingest ↔ features import cycles)."""

from __future__ import annotations

VIX_SPOT_SYMBOLS = frozenset({"VIX", "^VIX"})

VOLATILITY_PROXY_HINT = (
    "TwelveData does not provide spot VIX or ^VIX on most plans. "
    "Use VIXY (or VXX) at interval 1day with twelvedata_exchange: CBOE in "
    "context_ingest, then: sparkles ingest -c … -s VIXY -i 1day"
)


def validate_volatility_proxy_symbol(symbol: str) -> None:
    """Raise with guidance when user requests unsupported spot VIX tickers."""
    if symbol.upper().lstrip("^") in VIX_SPOT_SYMBOLS:
        raise ValueError(VOLATILITY_PROXY_HINT)
