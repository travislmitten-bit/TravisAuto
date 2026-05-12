"""
regime.py — Market regime detection using 200-day MA.
PARABOLIC : BTC >30% above 200-day MA (price extended, reduce hold fraction)
BULL      : BTC above 200-day MA for 30+ consecutive days
BEAR      : BTC below 200-day MA for 30+ consecutive days
CHOPPY    : Everything else
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
import numpy as np


class Regime(Enum):
    BULL      = "BULL"
    BEAR      = "BEAR"
    CHOPPY    = "CHOPPY"
    PARABOLIC = "PARABOLIC"


@dataclass
class RegimeParams:
    trail_mult:     float      # multiplier for ATR trailing stop distance
    max_risk_pct:   float      # max risk fraction per trade
    min_confidence: float      # minimum signal confidence to enter
    min_touches:    int        # minimum trendline touches required
    size_scalar:    float      # overall position size multiplier
    hold_pct:       float      # fraction to HOLD after partial TP exit
                               # exit_pct = 1 - hold_pct


REGIME_PARAMS: dict = {
    Regime.BULL:      RegimeParams(
        trail_mult=2.0, max_risk_pct=0.015, min_confidence=0.65,
        min_touches=2, size_scalar=1.0, hold_pct=0.40,   # 60/40 exit/hold
    ),
    Regime.BEAR:      RegimeParams(
        trail_mult=1.0, max_risk_pct=0.005, min_confidence=0.85,
        min_touches=4, size_scalar=0.5, hold_pct=0.20,   # 80/20 exit/hold
    ),
    Regime.CHOPPY:    RegimeParams(
        trail_mult=1.0, max_risk_pct=0.010, min_confidence=0.50,
        min_touches=5, size_scalar=0.5, hold_pct=0.25,   # 75/25 exit/hold
    ),
    Regime.PARABOLIC: RegimeParams(
        trail_mult=2.5, max_risk_pct=0.020, min_confidence=0.45,
        min_touches=2, size_scalar=1.2, hold_pct=0.50,   # 50/50 exit/hold
    ),
}


def detect_hh_hl_structure(
    highs_4h: list,
    lows_4h:  list,
    lookback: int = 10,
) -> bool:
    """
    Returns True when the last `lookback` consecutive 4H candles each made
    a higher high AND higher low than the prior candle.
    Qualifies the asset as BULL regardless of 200-day MA position.
    """
    n = len(highs_4h)
    if n < lookback + 1:
        return False
    h = highs_4h[-(lookback + 1):]
    l = lows_4h[-(lookback + 1):]
    for i in range(1, lookback + 1):
        if h[i] <= h[i - 1] or l[i] <= l[i - 1]:
            return False
    return True


def detect_responsive_bull(closes_daily: list) -> bool:
    """
    Three-condition fast-response BULL detector — fires regardless of 200-day MA:
      1. Current close > 50-day SMA
      2. 50-day SMA strictly rising for each of the last 5 days
         (SMA50_today > SMA50_yesterday > ... > SMA50_5d_ago)
      3. Higher high in the last 7 closes vs the prior 7 closes
         (max(closes[-7:]) > max(closes[-14:-7]))

    Designed to detect a confirmed bull leg early — before the 200-day MA
    catches up — so the manual XBTUSD_REGIME_OVERRIDE can be retired.
    """
    arr = closes_daily
    n = len(arr)
    # Need 50 candles for SMA + 5 historical SMAs (so prices 50+5 deep) + 14 for HH check.
    if n < 55 or n < 14:
        return False

    # Compute SMA50 today and 5 days back (6 points total).
    smas = []
    for offset in range(6):
        end = n - offset
        start = end - 50
        if start < 0:
            return False
        smas.append(sum(arr[start:end]) / 50.0)

    # (1) price > SMA50
    if arr[-1] <= smas[0]:
        return False
    # (2) SMA50 strictly rising each of the last 5 days
    for j in range(5):
        if smas[j] <= smas[j + 1]:
            return False
    # (3) HH in last 7 closes vs prior 7
    if max(arr[-7:]) <= max(arr[-14:-7]):
        return False
    return True


def detect_ma_uptrend(closes_daily: list) -> bool:
    """
    Returns True when all four uptrend conditions hold:
      1. price > SMA50
      2. SMA30 > SMA50         (golden-cross orientation — short MA above long MA)
      3. price[-1] > price[-7]  (weekly momentum)
      4. price[-7] > price[-14] (prior-week momentum)
    Requires ≥ 51 daily closes.
    """
    if len(closes_daily) < 51:
        return False
    arr = closes_daily
    sma50 = sum(arr[-50:]) / 50.0
    sma30 = sum(arr[-30:]) / 30.0
    return (
        arr[-1]  > sma50
        and sma30 > sma50
        and arr[-1]  > arr[-7]
        and arr[-7]  > arr[-14]
    )


def detect_regime(
    closes_daily: list,
    threshold: int = 10,
) -> Tuple[Regime, RegimeParams]:
    """
    Requires 200+ daily closes. Returns (Regime, RegimeParams).
    Counts consecutive daily closes above/below the 200-day SMA from most
    recent close backward. 10+ consecutive above = BULL (or PARABOLIC if
    >30% extended). 10+ consecutive below = BEAR. Otherwise CHOPPY.
    """
    arr = np.array(closes_daily, dtype=float)
    if len(arr) < 201:
        return Regime.CHOPPY, REGIME_PARAMS[Regime.CHOPPY]

    sma_200 = float(arr[-200:].mean())
    current = float(arr[-1])

    consecutive_above = 0
    consecutive_below = 0
    for price in arr[::-1]:
        if price > sma_200:
            if consecutive_below > 0:
                break
            consecutive_above += 1
        else:
            if consecutive_above > 0:
                break
            consecutive_below += 1

    if consecutive_above >= threshold:
        if current > sma_200 * 1.3:
            regime = Regime.PARABOLIC
        else:
            regime = Regime.BULL
    elif consecutive_below >= threshold:
        regime = Regime.BEAR
    else:
        regime = Regime.CHOPPY

    return regime, REGIME_PARAMS[regime]
