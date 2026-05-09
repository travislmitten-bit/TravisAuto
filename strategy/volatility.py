"""
volatility.py — Bollinger Band compression detection and pair correlation utilities.

VolatilityTracker:
  - Tracks per-pair BB width over a rolling 30-day window (180 four-hour readings)
  - Flags an asset PRIMED when its current BB width is the lowest in 30 days
  - PRIMED assets get +50% position size on the next signal, then reset

pearson_correlation:
  - Used by main.py to check correlation between a new position and all open ones
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Kraken pair → common symbol (for LunarCrush / external APIs)
PAIR_TO_SYMBOL: Dict[str, str] = {
    "XBTUSD": "BTC",
    "SOLUSD": "SOL",
    "TAOUSD": "TAO",
    "LINKUSD": "LINK",
}


# ── Bollinger Band width ──────────────────────────────────────────────────────

def calculate_bollinger_width(
    closes: List[float], period: int = 20, mult: float = 2.0
) -> float:
    """
    Normalised Bollinger Band width: (2 × mult × σ) / SMA.
    Returns 0.0 if fewer than 5 data points.
    """
    arr = np.array(closes[-period:] if len(closes) >= period else closes, dtype=float)
    if len(arr) < 5:
        return 0.0
    mid = arr.mean()
    if mid < 1e-9:
        return 0.0
    return float(2.0 * mult * arr.std() / mid)


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_correlation(a: List[float], b: List[float], n: int = 180) -> float:
    """
    Pearson correlation of the last n values of two price series.
    Returns 0.0 if fewer than 10 common points or if either series is flat.
    """
    arr_a = np.array(a[-n:], dtype=float)
    arr_b = np.array(b[-n:], dtype=float)
    k = min(len(arr_a), len(arr_b))
    if k < 10:
        return 0.0
    arr_a, arr_b = arr_a[-k:], arr_b[-k:]
    if arr_a.std() < 1e-9 or arr_b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


# ── Volatility Tracker ────────────────────────────────────────────────────────

class VolatilityTracker:
    """
    Per-asset Bollinger Band width history + PRIMED breakout state.

    An asset is PRIMED when its current 4h BB width equals the rolling
    30-day minimum — the classic volatility-compression-before-expansion setup.
    The PRIMED flag is consumed (reset) after the first signal on that asset.
    """

    _WINDOW   = 180   # 30 days × 6 four-hour candles
    _MIN_DATA = 30    # ≥ 5 days of readings before detection activates

    def __init__(self):
        self._history: Dict[str, Deque[float]] = {}
        self._primed:  Dict[str, bool]         = {}

    def update(self, pair: str, closes: List[float]) -> bool:
        """
        Append current BB width to history.
        Returns True if the asset just became PRIMED this call (state change).
        """
        width = calculate_bollinger_width(closes)
        if width <= 0:
            return False

        if pair not in self._history:
            self._history[pair] = deque(maxlen=self._WINDOW)
            self._primed[pair]  = False

        hist = self._history[pair]
        hist.append(width)

        if len(hist) < self._MIN_DATA:
            return False

        was_primed = self._primed[pair]
        compressed = width <= min(hist)   # current is the 30-day low
        self._primed[pair] = compressed

        if compressed and not was_primed:
            logger.warning(
                "BB COMPRESSION | %s | width=%.5f (30-day low across %d readings) | "
                "PRIMED — next signal gets +50%% size",
                pair, width, len(hist),
            )
            return True
        return False

    def is_primed(self, pair: str) -> bool:
        return self._primed.get(pair, False)

    def reset(self, pair: str):
        """Consume PRIMED status after a signal fires."""
        if self._primed.get(pair):
            logger.info("BB PRIMED consumed | %s | volatility scalar reset to 1.0", pair)
        self._primed[pair] = False

    def size_scalar(self, pair: str) -> float:
        return 1.5 if self.is_primed(pair) else 1.0
