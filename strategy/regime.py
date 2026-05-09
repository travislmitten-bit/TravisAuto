"""
regime.py — Market regime detection using 200-day MA.
BULL  : BTC above 200-day MA for 30+ consecutive days
BEAR  : BTC below 200-day MA for 30+ consecutive days
CHOPPY: Everything else
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np


class Regime(Enum):
    BULL   = "BULL"
    BEAR   = "BEAR"
    CHOPPY = "CHOPPY"


@dataclass
class RegimeParams:
    trail_mult: float      # multiplier for ATR trailing stop distance
    max_risk_pct: float    # max risk fraction per trade
    min_confidence: float  # minimum signal confidence to enter
    min_touches: int       # minimum trendline touches required
    size_scalar: float     # overall position size multiplier


REGIME_PARAMS: dict = {
    Regime.BULL:   RegimeParams(trail_mult=2.0, max_risk_pct=0.015, min_confidence=0.50, min_touches=2, size_scalar=1.0),
    Regime.BEAR:   RegimeParams(trail_mult=1.0, max_risk_pct=0.005, min_confidence=0.85, min_touches=2, size_scalar=0.5),
    Regime.CHOPPY: RegimeParams(trail_mult=1.0, max_risk_pct=0.010, min_confidence=0.50, min_touches=5, size_scalar=0.5),
}


def detect_regime(
    closes_daily: list,
    threshold: int = 30,
) -> Tuple[Regime, RegimeParams]:
    """
    Requires 200+ daily closes. Returns (Regime, RegimeParams).
    Uses a fixed 200-day SMA snapshot and checks how many of the last
    `threshold` daily closes sat above/below it.
    """
    arr = np.array(closes_daily, dtype=float)
    if len(arr) < 201:
        return Regime.CHOPPY, REGIME_PARAMS[Regime.CHOPPY]

    sma_200 = float(arr[-200:].mean())
    window  = arr[-threshold:]
    above   = int((window > sma_200).sum())

    if above == threshold:
        regime = Regime.BULL
    elif above == 0:
        regime = Regime.BEAR
    else:
        regime = Regime.CHOPPY

    return regime, REGIME_PARAMS[regime]
