"""
V8 Elite entry confirmation layer.
All calculations use existing Kraken OHLCV data — no additional API calls.

Exports:
  calculate_obv, calculate_rsi, calculate_macd      → raw series
  calculate_ichimoku, detect_wyckoff_spring          → structured results
  detect_hidden_divergence                           → shared for OBV + RSI
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ema_series(values: List[float], period: int) -> List[float]:
    """Full EMA series; first (period-1) elements are NaN."""
    n = len(values)
    if n < period:
        return [float("nan")] * n
    k   = 2.0 / (period + 1)
    out = [float("nan")] * n
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def _pivot_lows(values: List[float], window: int = 5) -> List[int]:
    lows = []
    for i in range(window, len(values) - window):
        seg = values[i - window: i + window + 1]
        if values[i] <= min(seg):
            lows.append(i)
    return lows


def _pivot_highs(values: List[float], window: int = 5) -> List[int]:
    highs = []
    for i in range(window, len(values) - window):
        seg = values[i - window: i + window + 1]
        if values[i] >= max(seg):
            highs.append(i)
    return highs


# ── OBV ───────────────────────────────────────────────────────────────────────

def calculate_obv(closes: List[float], volumes: List[float]) -> List[float]:
    """On-Balance Volume cumulative series."""
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


# ── RSI (Wilder smoothing) ────────────────────────────────────────────────────

def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """Returns RSI series (length = len(closes)); first (period+1) values are NaN."""
    n = len(closes)
    if n < period + 1:
        return [float("nan")] * n
    gains  = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, n)]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, n)]
    avg_g  = sum(gains[:period])  / period
    avg_l  = sum(losses[:period]) / period
    rsi: List[float] = [float("nan")] * (period + 1)
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rsi.append(100.0 if avg_l < 1e-10 else 100.0 - 100.0 / (1.0 + avg_g / avg_l))
    return rsi


# ── Hidden divergence (shared for OBV and RSI) ───────────────────────────────

@dataclass
class DivergenceResult:
    hidden_bull: bool = False   # price higher low  + indicator lower low  → continuation
    hidden_bear: bool = False   # price lower high  + indicator higher high → exhaustion


def detect_hidden_divergence(
    closes:    List[float],
    indicator: List[float],
    lookback:     int = 60,
    swing_window: int = 5,
) -> DivergenceResult:
    """
    Detects hidden divergence between price and a momentum/volume indicator.
    Only examines the most recent `lookback` candles.
    NaN values in indicator are skipped automatically.
    """
    if len(closes) < lookback or len(indicator) < lookback:
        return DivergenceResult()

    c   = closes[-lookback:]
    ind = indicator[-lookback:]

    valid = [i for i in range(len(c))
             if not (isinstance(ind[i], float) and np.isnan(ind[i]))]
    if len(valid) < swing_window * 2 + 2:
        return DivergenceResult()

    c_v   = [c[i]   for i in valid]
    ind_v = [ind[i] for i in valid]

    lows  = _pivot_lows(c_v,  swing_window)
    highs = _pivot_highs(c_v, swing_window)

    result = DivergenceResult()

    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if c_v[l2] > c_v[l1] and ind_v[l2] < ind_v[l1]:
            result.hidden_bull = True

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if c_v[h2] < c_v[h1] and ind_v[h2] > ind_v[h1]:
            result.hidden_bear = True

    return result


# ── MACD ─────────────────────────────────────────────────────────────────────

@dataclass
class MACDResult:
    histogram_last: float = 0.0
    histogram_prev: float = 0.0
    expanding:   bool = False   # hist[-1] > hist[-2] AND hist[-1] > 0
    contracting: bool = False   # hist[-1] < hist[-2] (momentum dying)


def calculate_macd(
    closes:        List[float],
    fast:          int = 12,
    slow:          int = 26,
    signal_period: int = 9,
) -> MACDResult:
    n = len(closes)
    if n < slow + signal_period:
        return MACDResult()

    ema_f = _ema_series(closes, fast)
    ema_s = _ema_series(closes, slow)
    macd_line = [
        (f - s) if not (np.isnan(f) or np.isnan(s)) else float("nan")
        for f, s in zip(ema_f, ema_s)
    ]

    valid_start = next((i for i, v in enumerate(macd_line) if not np.isnan(v)), None)
    if valid_start is None:
        return MACDResult()

    sig_vals = _ema_series(macd_line[valid_start:], signal_period)

    histogram = [float("nan")] * n
    for i, sig in enumerate(sig_vals):
        if not np.isnan(sig):
            idx = valid_start + i
            if idx < n:
                histogram[idx] = macd_line[idx] - sig

    valid_hist = [v for v in histogram if not np.isnan(v)]
    if len(valid_hist) < 2:
        return MACDResult()

    curr = valid_hist[-1]
    prev = valid_hist[-2]
    return MACDResult(
        histogram_last=curr,
        histogram_prev=prev,
        expanding=bool(curr > prev and curr > 0),
        contracting=bool(curr < prev),
    )


# ── Ichimoku Cloud ────────────────────────────────────────────────────────────

@dataclass
class IchimokuResult:
    tenkan:       Optional[float] = None
    kijun:        Optional[float] = None
    senkou_a:     Optional[float] = None   # Span A at current candle
    senkou_b:     Optional[float] = None   # Span B at current candle
    cloud_top:    Optional[float] = None
    cloud_bot:    Optional[float] = None
    above_cloud:  bool = False
    below_cloud:  bool = False
    tk_cross_bull: bool = False            # Tenkan just crossed above Kijun while above cloud


def calculate_ichimoku(
    highs:  List[float],
    lows:   List[float],
    closes: List[float],
    tenkan_period:   int = 9,
    kijun_period:    int = 26,
    senkou_b_period: int = 52,
    displacement:    int = 26,
) -> IchimokuResult:
    n = len(closes)

    def midpoint(period: int, idx: int) -> Optional[float]:
        if idx < period - 1 or idx >= n:
            return None
        return (max(highs[idx - period + 1: idx + 1])
                + min(lows[idx - period + 1: idx + 1])) / 2.0

    tenkan_now  = midpoint(tenkan_period, n - 1)
    kijun_now   = midpoint(kijun_period,  n - 1)
    tenkan_prev = midpoint(tenkan_period, n - 2)
    kijun_prev  = midpoint(kijun_period,  n - 2)

    # Cloud at current candle = Senkou A/B projected forward from (n-1-displacement)
    disp_idx = n - 1 - displacement
    senkou_a: Optional[float] = None
    senkou_b: Optional[float] = None
    if disp_idx >= 0:
        t_d = midpoint(tenkan_period,   disp_idx)
        k_d = midpoint(kijun_period,    disp_idx)
        b_d = midpoint(senkou_b_period, disp_idx)
        if t_d is not None and k_d is not None:
            senkou_a = (t_d + k_d) / 2.0
        senkou_b = b_d

    cloud_vals = [v for v in (senkou_a, senkou_b) if v is not None]
    cloud_top  = max(cloud_vals) if cloud_vals else None
    cloud_bot  = min(cloud_vals) if cloud_vals else None

    price       = closes[-1]
    above_cloud = cloud_top is not None and price > cloud_top
    below_cloud = cloud_bot is not None and price < cloud_bot

    tk_cross_bull = False
    if all(v is not None for v in [tenkan_now, kijun_now, tenkan_prev, kijun_prev]):
        crossed_up    = (tenkan_prev <= kijun_prev) and (tenkan_now > kijun_now)
        tk_cross_bull = crossed_up and above_cloud

    return IchimokuResult(
        tenkan=tenkan_now, kijun=kijun_now,
        senkou_a=senkou_a, senkou_b=senkou_b,
        cloud_top=cloud_top, cloud_bot=cloud_bot,
        above_cloud=above_cloud, below_cloud=below_cloud,
        tk_cross_bull=tk_cross_bull,
    )


# ── Wyckoff Spring ────────────────────────────────────────────────────────────

@dataclass
class WyckoffResult:
    detected:        bool  = False
    support_level:   float = 0.0
    penetration_pct: float = 0.0   # how far price dipped below support (%)
    vol_ratio:       float = 0.0   # bounce candle volume / average volume


def detect_wyckoff_spring(
    highs:   List[float],
    lows:    List[float],
    closes:  List[float],
    volumes: List[float],
    lookback:        int   = 40,
    max_penetration: float = 0.005,  # 0.5% breach max (was 1.5%)
    min_vol_ratio:   float = 2.0,    # 2x avg volume on bounce (was 1x)
    recovery_bars:   int   = 2,      # 2 candles of recovery (was 1)
    min_prior_tests: int   = 3,      # support must be tested ≥ 3 times before the spring
    test_tolerance:  float = 0.005,  # ±0.5% counts as a prior support test
) -> WyckoffResult:
    """
    Tightened Wyckoff spring:
      • support = highest-volume close in lookback window (ex. last `recovery_bars+1`)
      • support must have been TOUCHED ≥ min_prior_tests times before the spring
      • breach low ≤ 0.5% below support
      • bounce candle's volume ≥ 2 × avg(lookback volume)
      • next `recovery_bars` candles close back ABOVE support (sustained recovery)
    Caller is responsible for any cooldown (48h per asset, etc.).
    """
    n = len(closes)
    min_required = lookback + recovery_bars + 1
    if n < min_required:
        return WyckoffResult()

    # Identify support level — highest-volume candle in lookback, excluding the
    # final (recovery_bars + 1) candles (those contain the candidate spring).
    end_excl = -(recovery_bars + 1)
    vol_win   = np.array(volumes[-(lookback + recovery_bars + 1): end_excl])
    close_win = np.array(closes[-(lookback + recovery_bars + 1): end_excl])
    if len(vol_win) == 0:
        return WyckoffResult()
    support = float(close_win[int(np.argmax(vol_win))])
    avg_vol = float(np.mean(np.array(volumes[-lookback:])))

    # Count prior tests of support BEFORE the candidate spring candle.
    band_lo = support * (1.0 - test_tolerance)
    band_hi = support * (1.0 + test_tolerance)
    prior_tests = sum(
        1 for low in lows[-lookback - recovery_bars - 1: end_excl]
        if band_lo <= low <= band_hi
    )
    if prior_tests < min_prior_tests:
        return WyckoffResult()

    # Candidate spring candle is at offset = -(recovery_bars + 1).
    spring_idx = n + end_excl  # absolute index of the breach candle
    if spring_idx < 0:
        return WyckoffResult()
    low_v   = float(lows[spring_idx])
    vol_v   = float(volumes[spring_idx])
    pen     = (support - low_v) / (support + 1e-9)

    if not (0.0 < pen <= max_penetration):
        return WyckoffResult()
    if vol_v < avg_vol * min_vol_ratio:
        return WyckoffResult()

    # Recovery: subsequent `recovery_bars` candles must each close above support.
    for k in range(1, recovery_bars + 1):
        idx = spring_idx + k
        if idx >= n or closes[idx] <= support:
            return WyckoffResult()

    return WyckoffResult(
        detected=True,
        support_level=support,
        penetration_pct=round(pen * 100, 3),
        vol_ratio=round(vol_v / (avg_vol + 1e-9), 2),
    )


# ── SMC Order Block ───────────────────────────────────────────────────────────

@dataclass
class OrderBlockResult:
    detected:    bool  = False
    ob_high:     float = 0.0
    ob_low:      float = 0.0
    price_at_ob: bool  = False


def detect_order_block(
    opens:   List[float],
    highs:   List[float],
    lows:    List[float],
    closes:  List[float],
    lookback:       int   = 96,
    ob_window:      int   = 12,
    move_threshold: float = 0.10,
) -> OrderBlockResult:
    """
    Last bearish candle before a 10%+ bullish move in `ob_window` candles.
    Price returning to that range = institutional demand → +0.10 conf.
    """
    n = len(closes)
    if n < ob_window + 2:
        return OrderBlockResult()

    current    = closes[-1]
    search_end = n - ob_window - 1

    for i in range(search_end, max(search_end - lookback, 0), -1):
        if closes[i] >= opens[i]:
            continue
        future_high = max(highs[i + 1: i + ob_window + 1])
        if future_high < closes[i] * (1.0 + move_threshold):
            continue
        return OrderBlockResult(
            detected=True,
            ob_high=opens[i],
            ob_low=closes[i],
            price_at_ob=closes[i] <= current <= opens[i],
        )

    return OrderBlockResult()


# ── Fair Value Gap (FVG) ──────────────────────────────────────────────────────

@dataclass
class FVGResult:
    nearest_bullish_fvg: Optional[Tuple[float, float]] = None  # (low, high) above price
    nearest_bearish_fvg: Optional[Tuple[float, float]] = None  # (low, high) below price


def detect_fvg(
    highs:    List[float],
    lows:     List[float],
    closes:   List[float],
    lookback: int = 30,
) -> FVGResult:
    """Bullish FVG: lows[N+2] > highs[N] — gap above price = TP magnet for longs."""
    n = len(closes)
    if n < 3:
        return FVGResult()

    current = closes[-1]
    start   = max(0, n - lookback)

    nearest_bull:     Optional[Tuple[float, float]] = None
    nearest_bull_dist = float("inf")
    nearest_bear:     Optional[Tuple[float, float]] = None
    nearest_bear_dist = float("inf")

    for i in range(start, n - 2):
        fvg_low  = highs[i]
        fvg_high = lows[i + 2]
        if fvg_high > fvg_low and fvg_low > current:
            dist = fvg_low - current
            if dist < nearest_bull_dist:
                nearest_bull_dist = dist
                nearest_bull = (fvg_low, fvg_high)

        bfvg_low  = highs[i + 2]
        bfvg_high = lows[i]
        if bfvg_high > bfvg_low and bfvg_high < current:
            dist = current - bfvg_high
            if dist < nearest_bear_dist:
                nearest_bear_dist = dist
                nearest_bear = (bfvg_low, bfvg_high)

    return FVGResult(nearest_bullish_fvg=nearest_bull, nearest_bearish_fvg=nearest_bear)


# ── Fibonacci Retracement & Extensions ───────────────────────────────────────

@dataclass
class FibResult:
    swing_low:    float = 0.0
    swing_high:   float = 0.0
    fib_618:      float = 0.0
    fib_ext_1272: float = 0.0
    fib_ext_1618: float = 0.0
    fib_ext_2618: float = 0.0
    at_618:       bool  = False  # price within 1% of 61.8% retracement


def calculate_fibonacci(
    highs:     List[float],
    lows:      List[float],
    closes:    List[float],
    lookback:  int   = 60,
    tolerance: float = 0.01,
) -> FibResult:
    """
    Fibonacci levels from most recent swing low/high within `lookback` candles.
    Extensions (127.2%, 161.8%, 261.8%) = TP targets; 61.8% = bounce confirmation.
    """
    n = len(closes)
    if n < lookback:
        return FibResult()

    swing_high = max(highs[-lookback:])
    swing_low  = min(lows[-lookback:])

    if swing_high <= swing_low:
        return FibResult()

    rng   = swing_high - swing_low
    price = closes[-1]

    fib_618      = swing_high - 0.618 * rng
    fib_ext_1272 = swing_low  + 1.272 * rng
    fib_ext_1618 = swing_low  + 1.618 * rng
    fib_ext_2618 = swing_low  + 2.618 * rng

    return FibResult(
        swing_low=round(swing_low, 8),
        swing_high=round(swing_high, 8),
        fib_618=round(fib_618, 8),
        fib_ext_1272=round(fib_ext_1272, 8),
        fib_ext_1618=round(fib_ext_1618, 8),
        fib_ext_2618=round(fib_ext_2618, 8),
        at_618=abs(price - fib_618) / (fib_618 + 1e-9) < tolerance,
    )
