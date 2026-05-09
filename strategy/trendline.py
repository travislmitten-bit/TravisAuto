"""
Tori's Trendline Strategy
--------------------------
1. Identify swing highs/lows over a rolling window.
2. Fit support and resistance trendlines through the pivots.
3. Signal on:
   - Bounce: price touches line (within tolerance) + rejection candle
   - Breakout: price closes beyond line by breakout_threshold
4. Volume filter: volume on signal candle > 1.2× 20-period average.
"""

from __future__ import annotations
import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from config.config import TrendlineConfig


class Signal(Enum):
    NONE = "none"
    BUY_BOUNCE = "buy_bounce"       # price bounced off support
    SELL_BOUNCE = "sell_bounce"     # price bounced off resistance
    BUY_BREAK = "buy_break"         # bullish breakout above resistance
    SELL_BREAK = "sell_break"       # bearish breakdown below support


@dataclass
class Trendline:
    slope: float
    intercept: float
    touches: List[int]              # candle indices of confirmed touches
    line_type: str                  # "support" or "resistance"
    start_idx: int
    end_idx: int
    strength: float = 0.0           # touch count weighted by recency

    def price_at(self, idx: int) -> float:
        return self.slope * idx + self.intercept

    def distance_pct(self, idx: int, price: float) -> float:
        line_price = self.price_at(idx)
        return (price - line_price) / line_price if line_price else 0.0

    @property
    def angle_degrees(self) -> float:
        return math.degrees(math.atan(self.slope))


@dataclass
class TrendlineSignal:
    signal: Signal
    pair: str
    price: float
    trendline: Trendline
    candle_idx: int
    confidence: float               # 0–1 based on touches + volume
    suggested_stop: float = 0.0
    suggested_target: float = 0.0
    extra: dict = field(default_factory=dict)


def _find_swing_highs(highs: np.ndarray, window: int) -> List[int]:
    pivots = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            pivots.append(i)
    return pivots


def _find_swing_lows(lows: np.ndarray, window: int) -> List[int]:
    pivots = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            pivots.append(i)
    return pivots


def _fit_line(x: List[int], y: List[float]) -> Tuple[float, float]:
    """Least-squares line through pivot points."""
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    if len(xs) < 2:
        return 0.0, ys[0] if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _count_touches(
    indices: List[int],
    prices: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
) -> List[int]:
    touches = []
    for i in indices:
        line_p = slope * i + intercept
        if line_p and abs(prices[i] - line_p) / line_p <= tolerance:
            touches.append(i)
    return touches


def _trendline_strength(touches: List[int], total_candles: int) -> float:
    """Weight recent touches more heavily."""
    if not touches:
        return 0.0
    score = sum((t / total_candles) for t in touches)
    return round(score / len(touches) * len(touches), 3)


def detect_trendlines(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    cfg: TrendlineConfig,
) -> Tuple[List[Trendline], List[Trendline]]:
    """Return (support_lines, resistance_lines) sorted by strength desc."""
    n = len(closes)
    swing_h = _find_swing_highs(highs, cfg.swing_window)
    swing_l = _find_swing_lows(lows, cfg.swing_window)

    def build_lines(pivot_idx: List[int], pivot_prices: np.ndarray, ltype: str) -> List[Trendline]:
        lines: List[Trendline] = []
        # Iterate over all pairs of pivots as seed points
        for i in range(len(pivot_idx)):
            for j in range(i + 1, len(pivot_idx)):
                x0, x1 = pivot_idx[i], pivot_idx[j]
                y0, y1 = pivot_prices[x0], pivot_prices[x1]
                if x1 == x0:
                    continue
                slope = (y1 - y0) / (x1 - x0)
                intercept = y0 - slope * x0

                angle = abs(math.degrees(math.atan(slope)))
                if angle < cfg.min_slope_angle:
                    continue

                touches = _count_touches(
                    pivot_idx, pivot_prices, slope, intercept, cfg.touch_tolerance
                )
                if len(touches) < cfg.min_touches:
                    continue

                strength = _trendline_strength(touches, n)
                lines.append(
                    Trendline(
                        slope=slope,
                        intercept=intercept,
                        touches=touches,
                        line_type=ltype,
                        start_idx=pivot_idx[i],
                        end_idx=pivot_idx[-1],
                        strength=strength,
                    )
                )
        # Deduplicate near-identical lines
        lines.sort(key=lambda l: l.strength, reverse=True)
        deduped: List[Trendline] = []
        for line in lines:
            duplicate = False
            for kept in deduped:
                if abs(line.slope - kept.slope) < 1e-6 and abs(line.intercept - kept.intercept) / (abs(kept.intercept) + 1e-9) < 0.001:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(line)
        return deduped

    support_lines = build_lines(swing_l, lows, "support")
    resistance_lines = build_lines(swing_h, highs, "resistance")
    return support_lines, resistance_lines


def _is_rejection_candle(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    line_type: str,
) -> bool:
    """Simple wick-rejection check."""
    body = abs(closes[idx] - opens[idx])
    total = highs[idx] - lows[idx]
    if total < 1e-9:
        return False
    body_ratio = body / total
    if line_type == "support":
        lower_wick = min(opens[idx], closes[idx]) - lows[idx]
        return lower_wick / total > 0.4 and closes[idx] > opens[idx]
    else:
        upper_wick = highs[idx] - max(opens[idx], closes[idx])
        return upper_wick / total > 0.4 and closes[idx] < opens[idx]


def generate_signals(
    pair: str,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    cfg: TrendlineConfig,
    rr_ratio: float = 2.0,
    min_touches_override: Optional[int] = None,
) -> List[TrendlineSignal]:
    n = len(closes)
    if n < cfg.lookback_candles:
        return []

    active_cfg = (
        dataclasses.replace(cfg, min_touches=min_touches_override)
        if min_touches_override is not None else cfg
    )
    support_lines, resistance_lines = detect_trendlines(opens, highs, lows, closes, active_cfg)

    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    current_vol = float(volumes[-1])
    vol_ok = current_vol >= avg_vol * 1.2

    signals: List[TrendlineSignal] = []
    last = n - 1
    price = float(closes[last])

    for line in support_lines:
        dist = line.distance_pct(last, price)

        # Bounce signal: price within touch tolerance of support, rejection candle
        if abs(dist) <= cfg.touch_tolerance:
            rejection = _is_rejection_candle(opens, closes, highs, lows, last, "support")
            if rejection and vol_ok:
                stop = float(lows[last]) * (1 - 0.002)
                target = price + (price - stop) * rr_ratio
                confidence = min(1.0, len(line.touches) / 5 * 0.7 + (0.3 if vol_ok else 0))
                signals.append(TrendlineSignal(
                    signal=Signal.BUY_BOUNCE,
                    pair=pair,
                    price=price,
                    trendline=line,
                    candle_idx=last,
                    confidence=round(confidence, 3),
                    suggested_stop=round(stop, 8),
                    suggested_target=round(target, 8),
                    extra={"volume_ratio": round(current_vol / avg_vol, 2)},
                ))

        # Breakdown: price closed below support by threshold
        if dist < -cfg.breakout_threshold and vol_ok:
            stop = line.price_at(last) * (1 + 0.003)
            target = price - (stop - price) * rr_ratio
            confidence = min(1.0, len(line.touches) / 5 * 0.7 + 0.3)
            signals.append(TrendlineSignal(
                signal=Signal.SELL_BREAK,
                pair=pair,
                price=price,
                trendline=line,
                candle_idx=last,
                confidence=round(confidence, 3),
                suggested_stop=round(stop, 8),
                suggested_target=round(target, 8),
                extra={"volume_ratio": round(current_vol / avg_vol, 2)},
            ))

    for line in resistance_lines:
        dist = line.distance_pct(last, price)

        # Bounce off resistance
        if abs(dist) <= cfg.touch_tolerance:
            rejection = _is_rejection_candle(opens, closes, highs, lows, last, "resistance")
            if rejection and vol_ok:
                stop = float(highs[last]) * (1 + 0.002)
                target = price - (stop - price) * rr_ratio
                confidence = min(1.0, len(line.touches) / 5 * 0.7 + (0.3 if vol_ok else 0))
                signals.append(TrendlineSignal(
                    signal=Signal.SELL_BOUNCE,
                    pair=pair,
                    price=price,
                    trendline=line,
                    candle_idx=last,
                    confidence=round(confidence, 3),
                    suggested_stop=round(stop, 8),
                    suggested_target=round(target, 8),
                    extra={"volume_ratio": round(current_vol / avg_vol, 2)},
                ))

        # Bullish breakout above resistance
        if dist > cfg.breakout_threshold and vol_ok:
            stop = line.price_at(last) * (1 - 0.003)
            target = price + (price - stop) * rr_ratio
            confidence = min(1.0, len(line.touches) / 5 * 0.7 + 0.3)
            signals.append(TrendlineSignal(
                signal=Signal.BUY_BREAK,
                pair=pair,
                price=price,
                trendline=line,
                candle_idx=last,
                confidence=round(confidence, 3),
                suggested_stop=round(stop, 8),
                suggested_target=round(target, 8),
                extra={"volume_ratio": round(current_vol / avg_vol, 2)},
            ))

    # Return highest-confidence signal per pair
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals
