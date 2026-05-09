"""
analytics.py — ADX, Kelly Criterion, Sharpe Ratio, and asset scoring.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── ADX ───────────────────────────────────────────────────────────────────────

def calculate_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _calculate_adx_inner(highs, lows, closes, period)


def _calculate_adx_inner(highs: list, lows: list, closes: list, period: int) -> float:
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    n = len(c)
    if n < period + 2:
        return 0.0

    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    mdm  = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        up     = h[i] - h[i-1]
        dn     = l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        mdm[i] = dn if dn > up and dn > 0 else 0.0

    def _wilder(arr: np.ndarray, p: int) -> np.ndarray:
        s = np.zeros(len(arr))
        s[p] = arr[1:p+1].sum()
        for i in range(p + 1, len(arr)):
            s[i] = s[i-1] - s[i-1] / p + arr[i]
        return s

    atr  = _wilder(tr, period)
    pdi  = np.where(atr > 0, 100 * _wilder(pdm, period) / atr, 0.0)
    mdi  = np.where(atr > 0, 100 * _wilder(mdm, period) / atr, 0.0)
    dsum = pdi + mdi
    dx   = np.where(dsum > 0, 100 * np.abs(pdi - mdi) / dsum, 0.0)
    adx  = _wilder(dx, period)
    return float(adx[-1])


def adx_scalars(adx_value: float) -> Tuple[bool, float, Optional[float]]:
    """
    Returns (skip_trade, size_scalar, risk_cap_override).
    skip_trade=True  → ADX too low, skip this signal entirely.
    risk_cap_override → when ADX >50, cap effective risk at 2%.
    """
    if adx_value < 20:
        return True, 0.0, None
    elif adx_value < 35:
        return False, 1.0, None
    elif adx_value < 50:
        return False, 1.5, None
    else:
        return False, 1.0, 0.02


# ── SMA ───────────────────────────────────────────────────────────────────────

def calculate_sma(closes: list, period: int) -> float:
    arr = np.array(closes, dtype=float)
    if len(arr) < period:
        return float(arr[-1]) if len(arr) else 0.0
    return float(arr[-period:].mean())


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def kelly_fraction(
    trade_history: List[Tuple[float, float]],
    max_kelly: float = 0.02,
) -> float:
    """
    trade_history: list of (pnl, trade_value) tuples (most recent last).
    Returns Kelly fraction [0, max_kelly], or 0 if fewer than 50 trades.
    """
    if len(trade_history) < 50:
        return 0.0
    recent = list(trade_history)[-50:]
    wins   = [(p, v) for p, v in recent if p > 0 and v > 0]
    losses = [(p, v) for p, v in recent if p <= 0 and v > 0]
    if not wins or not losses:
        return max_kelly / 2

    win_rate = len(wins) / len(recent)
    avg_win  = float(np.mean([p / v for p, v in wins]))
    avg_loss = float(np.mean([abs(p) / v for p, v in losses]))
    if avg_loss < 1e-9:
        return max_kelly

    b     = avg_win / avg_loss
    kelly = (b * win_rate - (1 - win_rate)) / b
    return float(max(0.0, min(kelly, max_kelly)))


# ── Sharpe Ratio ──────────────────────────────────────────────────────────────

def rolling_sharpe(daily_pnl_pcts: List[float], period: int = 30) -> float:
    """Annualised Sharpe on a rolling window of daily PnL percentages."""
    if len(daily_pnl_pcts) < 5:
        return 0.0
    arr = np.array(daily_pnl_pcts[-period:], dtype=float)
    if arr.std() < 1e-9:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


# ── Asset rotation scoring ────────────────────────────────────────────────────

def score_asset(closes: list, volumes: list) -> float:
    """
    Simple momentum + volume trend score for weekly rotation.
    Higher score → allocate more capital.
    """
    c = np.array(closes[-60:], dtype=float)
    v = np.array(volumes[-60:], dtype=float)
    if len(c) < 21:
        return 0.0
    mom_5   = (c[-1] - c[-6])  / (c[-6]  + 1e-9) if len(c) >= 6  else 0.0
    mom_20  = (c[-1] - c[-21]) / (c[-21] + 1e-9) if len(c) >= 21 else 0.0
    sma_20  = c[-20:].mean()
    trend   = (c[-1] - sma_20) / (sma_20 + 1e-9)
    vol_r   = v[-5:].mean() / (v[-20:].mean() + 1e-9) - 1.0
    return float(0.35 * mom_5 + 0.25 * mom_20 + 0.20 * trend + 0.20 * vol_r)


def compute_rotation_scalars(pair_scores: Dict[str, float]) -> Dict[str, float]:
    """Top-ranked asset gets 1.3×, bottom gets 0.7×, middle 1.0×."""
    if not pair_scores:
        return {}
    ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)
    scalars: Dict[str, float] = {}
    for i, (pair, _) in enumerate(ranked):
        if n == 1:
            scalars[pair] = 1.0
        elif i == 0:
            scalars[pair] = 1.3
        elif i == n - 1:
            scalars[pair] = 0.7
        else:
            scalars[pair] = 1.0
    return scalars
