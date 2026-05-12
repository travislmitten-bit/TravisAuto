"""
ema_ribbon.py — 8-period EMA ribbon for entry confirmation, trailing exit tightening,
and composite strength scoring.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144, 233]

_COMPRESSION_THRESHOLD = 0.02   # all EMAs within 2% of each other → COMPRESSED
_SPREAD_FANNING_MIN    = 0.005  # ≥0.5% spread required to count as fanning


class RibbonState(Enum):
    BULL_FANNING = "BULL_FANNING"
    BEAR_FANNING = "BEAR_FANNING"
    COMPRESSED   = "COMPRESSED"
    MIXED        = "MIXED"


@dataclass
class RibbonResult:
    state:         RibbonState
    strength:      float           # 0-100
    emas:          Dict[int, float] # period → current EMA value
    composite_pts: float           # -10 to +10 contribution to composite score


def calculate_ema(closes: List[float], period: int) -> float:
    """Standard EMA seeded from SMA of the first `period` values."""
    arr = np.array(closes, dtype=float)
    if len(arr) < period:
        return float(arr[-1]) if len(arr) else 0.0
    k   = 2.0 / (period + 1)
    ema = float(arr[:period].mean())
    for price in arr[period:]:
        ema = price * k + ema * (1.0 - k)
    return float(ema)


def calculate_ribbon(closes: List[float]) -> Dict[int, float]:
    """Return {period: ema_value} for every ribbon period."""
    return {p: calculate_ema(closes, p) for p in EMA_PERIODS}


def ribbon_strength_score(emas: Dict[int, float]) -> float:
    """
    0-100 based on two components:
      60 pts — ordering: fraction of consecutive EMA pairs in bull or bear order
      40 pts — spread:   (max-min)/min normalised against a 5% reference spread
    """
    vals    = [emas[p] for p in EMA_PERIODS]
    n_pairs = len(vals) - 1

    bull_ok = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
    bear_ok = sum(1 for a, b in zip(vals, vals[1:]) if a < b)
    order_score = max(bull_ok, bear_ok) / n_pairs

    lo      = min(vals)
    hi      = max(vals)
    spread  = (hi - lo) / (lo + 1e-9)
    spread_score = min(1.0, spread / 0.05)

    return round(60.0 * order_score + 40.0 * spread_score, 1)


def _detect_state(emas: Dict[int, float]) -> RibbonState:
    vals    = [emas[p] for p in EMA_PERIODS]
    lo      = min(vals)
    hi      = max(vals)
    spread  = (hi - lo) / (lo + 1e-9)

    if spread < _COMPRESSION_THRESHOLD:
        return RibbonState.COMPRESSED

    n_pairs = len(vals) - 1
    bull_ok = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
    bear_ok = sum(1 for a, b in zip(vals, vals[1:]) if a < b)

    if bull_ok == n_pairs and spread >= _SPREAD_FANNING_MIN:
        return RibbonState.BULL_FANNING
    if bear_ok == n_pairs and spread >= _SPREAD_FANNING_MIN:
        return RibbonState.BEAR_FANNING
    return RibbonState.MIXED


def ribbon_composite_pts(strength: float) -> float:
    """-10 to +10 pts; 50 strength = neutral (0 pts)."""
    return round((strength - 50.0) / 5.0, 2)


def evaluate_ribbon(closes: List[float]) -> RibbonResult:
    emas     = calculate_ribbon(closes)
    state    = _detect_state(emas)
    strength = ribbon_strength_score(emas)
    pts      = ribbon_composite_pts(strength)
    return RibbonResult(state=state, strength=strength, emas=emas, composite_pts=pts)
