"""
tests/test_all_systems.py — Comprehensive system test suite for TravisAuto.

Run from the repo root:
    python3 tests/test_all_systems.py

No external test dependencies. Each test prints PASS/FAIL with a short reason.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force dry-run mode so the bot construction doesn't try to hit Kraken private endpoints.
os.environ.setdefault("DRY_RUN", "true")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import logging
import math
import numpy as np

from config.config import CONFIG, TrendlineConfig, RiskConfig
from strategy.trendline import detect_trendlines
from strategy.ema_ribbon import (
    calculate_ema, calculate_ribbon, evaluate_ribbon, RibbonState, EMA_PERIODS,
)
from strategy.indicators import (
    calculate_obv, calculate_rsi, calculate_macd, calculate_ichimoku,
)
from strategy.analytics import kelly_fraction
from strategy.regime import (
    Regime, REGIME_PARAMS, detect_responsive_bull, detect_ma_uptrend,
)
from strategy.data_scanner import DataScanner
from strategy.risk_management import RiskManager


# ── Fixtures ────────────────────────────────────────────────────────────────

def _trending_ohlcv(n: int = 100):
    """
    Build a clean uptrending OHLCV series where swing-lows lie exactly on the
    line  y = 1.0 * x + 98  (i.e. base price = 100 + 1*i, swing-low at base - 2).
    Lows at i ∈ {10, 25, 40, 55, 70, 85} are clear local minima within ±5.
    """
    opens   = np.zeros(n)
    highs   = np.zeros(n)
    lows    = np.zeros(n)
    closes  = np.zeros(n)
    volumes = np.zeros(n) + 1000.0

    low_anchors = {10, 25, 40, 55, 70, 85}
    for i in range(n):
        base = 100.0 + 1.0 * i
        if i in low_anchors:
            low  = base - 2.0          # swing low touches support line (y = x + 98)
            close = base + 1.0
            high  = base + 2.0
        else:
            low   = base + 5.0
            close = base + 7.0
            high  = base + 9.0
        opens[i]  = close - 0.3
        closes[i] = close
        highs[i]  = high
        lows[i]   = low
        volumes[i] = 1000 + i * 5  # rising volume for trendline volume filter

    return opens, highs, lows, closes, volumes


def _linear_ramp(n: int):
    """Returns closes = [1.0, 2.0, ..., n] — used for EMA/RSI determinism."""
    return [float(i + 1) for i in range(n)]


# ── Tests ───────────────────────────────────────────────────────────────────

def test_01_trendline():
    opens, highs, lows, closes, _ = _trending_ohlcv(100)
    cfg = TrendlineConfig()
    supports, _resistances = detect_trendlines(opens, highs, lows, closes, cfg)
    assert supports, "no support trendlines detected"
    qualifying = [s for s in supports if len(s.touches) >= 4 and s.slope > 0]
    assert qualifying, (
        f"no support line with ≥4 touches and positive slope "
        f"(found {len(supports)} lines, max touches = "
        f"{max(len(s.touches) for s in supports) if supports else 0})"
    )
    best = qualifying[0]
    assert best.slope > 0, f"slope not positive: {best.slope}"
    # Sanity: angle should be above the 5° min_slope filter
    assert abs(math.degrees(math.atan(best.slope))) > 5.0


def test_02_ema_ribbon():
    # Known-value check: EMA(10) of [1..100] approaches N - (period-1)/2 = 95.5
    small = _linear_ramp(100)
    ema10 = calculate_ema(small, 10)
    assert abs(ema10 - 95.5) < 0.5, f"EMA(10) of 1..100 expected ~95.5, got {ema10:.3f}"

    # All 8 ribbon periods present and ordered correctly in an uptrend (300 candles)
    closes = _linear_ramp(300)
    emas = calculate_ribbon(closes)
    assert sorted(emas.keys()) == sorted(EMA_PERIODS), f"missing periods: {emas.keys()}"
    vals = [emas[p] for p in EMA_PERIODS]   # period ascending: 8,13,21,...
    # In a steady uptrend, shorter EMA > longer EMA at every step
    for short, long in zip(EMA_PERIODS, EMA_PERIODS[1:]):
        assert emas[short] > emas[long], (
            f"EMA ordering broken: EMA({short})={emas[short]:.3f} ≤ EMA({long})={emas[long]:.3f}"
        )
    result = evaluate_ribbon(closes)
    assert result.state == RibbonState.BULL_FANNING, (
        f"expected BULL_FANNING in linear uptrend, got {result.state}"
    )


def test_03_vwap():
    """Manual VWAP across 6 candles (the window used by main.py)."""
    highs   = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    lows    = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    closes  = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
    volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    # tp = (h+l+c)/3 per candle
    tp = (highs + lows + closes) / 3.0
    expected = float((tp * volumes).sum() / volumes.sum())

    # Import after sys.path is set; main.py boots logging which writes to logs/
    from main import TravisAutoBot
    data = {"highs": highs, "lows": lows, "closes": closes, "volumes": volumes}
    got = TravisAutoBot._calculate_daily_vwap(data)
    assert abs(got - expected) < 1e-6, f"VWAP mismatch: expected {expected}, got {got}"


def test_04_macd():
    # Compounding uptrend (0.5% per step) → strictly accelerating in absolute price increments
    closes = [100.0 * (1.005 ** i) for i in range(200)]
    res = calculate_macd(closes, fast=12, slow=26, signal_period=9)
    assert res.histogram_last != 0.0, "MACD histogram is zero — calc failed"
    assert res.histogram_last > 0, f"hist should be > 0 in accelerating uptrend, got {res.histogram_last:.4f}"
    assert res.expanding is True, (
        f"expected expanding=True in compounding uptrend "
        f"(hist {res.histogram_prev:.4f} → {res.histogram_last:.4f})"
    )

    # Strong uptrend then sharp reversal: MACD line plunges, signal lags → hist contracting at end
    closes_decel = [100.0 + 0.8 * i for i in range(195)]
    last = closes_decel[-1]
    closes_decel += [last - 2.0 * k for k in range(1, 6)]   # 5 sharp-down candles
    res2 = calculate_macd(closes_decel)
    assert res2.contracting is True, (
        f"expected contracting=True after sharp reversal "
        f"(hist {res2.histogram_prev:.4f} → {res2.histogram_last:.4f})"
    )


def test_05_ichimoku():
    # 200-candle clean uptrend
    closes = _linear_ramp(200)
    highs  = [c + 1.0 for c in closes]
    lows   = [c - 1.0 for c in closes]
    res = calculate_ichimoku(highs, lows, closes)
    # All 5 components present
    for field in ("tenkan", "kijun", "senkou_a", "senkou_b"):
        assert getattr(res, field) is not None, f"{field} missing"
    assert res.cloud_top is not None and res.cloud_bot is not None
    # In a sustained uptrend, current price (200) is far above the cloud which is
    # projected from 26 candles back
    assert res.above_cloud is True, "expected above_cloud=True in linear uptrend"
    assert res.below_cloud is False


def test_06_obv():
    # Manual OBV verification
    closes  = [100.0, 101.0, 102.0, 101.0, 100.0]
    volumes = [10.0,  20.0,  30.0,  40.0,  50.0]
    obv = calculate_obv(closes, volumes)
    # OBV[0] = 0
    # 101>100 → +20  → 20
    # 102>101 → +30  → 50
    # 101<102 → -40  → 10
    # 100<101 → -50  → -40
    expected = [0.0, 20.0, 50.0, 10.0, -40.0]
    assert obv == expected, f"OBV mismatch: expected {expected}, got {obv}"

    # Extend to 50 candles with mixed pattern — just verify it accumulates without error
    closes_50 = [100.0 + math.sin(i / 3.0) * 5 for i in range(50)]
    vols_50   = [1000.0] * 50
    obv_50 = calculate_obv(closes_50, vols_50)
    assert len(obv_50) == 50


def test_07_rsi():
    # Monotonic uptrend: all gains, no losses → RSI should saturate at 100
    closes = _linear_ramp(50)
    rsi = calculate_rsi(closes, period=14)
    assert len(rsi) == 50
    # Last value must be defined and equal 100 (within tolerance)
    last = rsi[-1]
    assert not (isinstance(last, float) and math.isnan(last)), "RSI tail is NaN"
    assert abs(last - 100.0) < 0.1, f"RSI on monotonic uptrend expected ≈100, got {last:.4f}"

    # Monotonic downtrend → RSI should saturate at 0
    closes_down = list(reversed(closes))
    rsi_down = calculate_rsi(closes_down, period=14)
    assert abs(rsi_down[-1] - 0.0) < 0.1, f"RSI on downtrend expected ≈0, got {rsi_down[-1]:.4f}"


def test_08_kelly():
    # Below the 50-trade activation threshold → 0
    assert kelly_fraction([(10.0, 1000.0)] * 49) == 0.0

    # 50 trades, 30 wins of $20 / 20 losses of $10, all trade_value = $1000
    #   win_rate = 0.6, avg_win = 0.02, avg_loss = 0.01, b = 2
    #   raw_kelly = (2*0.6 - 0.4) / 2 = 0.4  — capped at max_kelly = 0.02
    hist = [(20.0, 1000.0)] * 30 + [(-10.0, 1000.0)] * 20
    k = kelly_fraction(hist, max_kelly=0.02)
    assert abs(k - 0.02) < 1e-9, f"expected capped Kelly = 0.02, got {k}"

    # All wins (no losses) → returns max_kelly/2 = 0.01 sentinel
    only_wins = [(10.0, 1000.0)] * 50
    assert kelly_fraction(only_wins, max_kelly=0.02) == 0.01

    # Loss-heavy: raw kelly < 0 → clipped to 0
    bad = [(5.0, 1000.0)] * 10 + [(-50.0, 1000.0)] * 40   # win_rate=0.2, avg_win=0.005, avg_loss=0.05 → b=0.1
    #   raw = (0.1 * 0.2 - 0.8) / 0.1 = (0.02 - 0.8) / 0.1 = -7.8 → clamp to 0
    assert kelly_fraction(bad, max_kelly=0.02) == 0.0


def test_09_composite_score():
    # Six contributing sources with known scores. Weights:
    # fg .25, fund .20, mom .20, on_chain .15, mempool .10, sol .10
    scores = {
        "fear_greed":      50.0,
        "funding_rate":    60.0,
        "momentum":        70.0,
        "on_chain_btc":    60.0,
        "mempool":         50.0,
        "solana_activity": 50.0,
    }
    expected = 50 * 0.25 + 60 * 0.20 + 70 * 0.20 + 60 * 0.15 + 50 * 0.10 + 50 * 0.10
    got = DataScanner._weighted_composite(scores)
    assert abs(got - expected) < 0.01, f"composite mismatch: expected {expected}, got {got}"
    assert 0.0 <= got <= 100.0, f"composite out of range: {got}"

    # All-None → defaults to neutral 50
    assert DataScanner._weighted_composite({k: None for k in scores}) == 50.0


def test_10_position_sizing():
    rm = RiskManager(RiskConfig(), history_path=None)
    # Balance $10,000, entry $100, stop $95 (per-unit risk $5), risk 1%
    # Expected volume = (10,000 * 0.01) / 5 = 20.0
    vol = rm.calculate_position_size(
        account_balance=10_000.0, entry=100.0, stop=95.0, effective_risk_pct=0.01,
    )
    assert abs(vol - 20.0) < 1e-9, f"expected 20.0, got {vol}"

    # Zero per-unit risk should return 0 (no division by zero)
    assert rm.calculate_position_size(10_000.0, 100.0, 100.0, 0.01) == 0.0


def test_11_bull_regime_params():
    params = REGIME_PARAMS[Regime.BULL]
    # BULL baseline tightened to 0.65 to filter weak signals while remaining aggressive.
    # CLARITY-Act catalyst (0.72) still raises this further during May 13–15.
    assert params.min_confidence == 0.65, (
        f"BULL min_confidence baseline mismatch: expected 0.65, got {params.min_confidence}"
    )
    assert params.size_scalar == 1.0, f"BULL size_scalar: expected 1.0, got {params.size_scalar}"
    assert params.trail_mult  == 2.0, f"BULL trail_mult: expected 2.0, got {params.trail_mult}"


def test_12_dynamic_split_bull():
    params = REGIME_PARAMS[Regime.BULL]
    # 60/40 means exit_pct = 0.60, hold_pct = 0.40
    assert abs(params.hold_pct - 0.40) < 1e-9, (
        f"BULL hold_pct expected 0.40 (60/40 split), got {params.hold_pct}"
    )
    exit_pct = 1.0 - params.hold_pct
    assert abs(exit_pct - 0.60) < 1e-9


def test_13_kill_switch():
    rm = RiskManager(RiskConfig(), history_path=None)
    rm.update_peak(100_000.0)
    # 5% drawdown → fine
    assert rm.can_trade(95_000.0), "5% drawdown should not gate trading"
    # 15.1% drawdown → pause, but not kill switch
    assert not rm.can_trade(84_900.0), "15%+ drawdown should pause new entries"
    assert not rm.kill_switch_active, "kill switch should not yet be armed at 15%"
    # 20% drawdown → kill switch arms
    assert not rm.can_trade(80_000.0)
    assert rm.kill_switch_active, "kill switch should be armed at 20% drawdown"
    # Even back at peak balance, kill switch stays armed (manual restart required)
    assert not rm.can_trade(100_000.0), "kill switch must NOT auto-clear at peak recovery"


def test_14_profit_threshold():
    entry      = 100.0
    target_low = 102.0   # 2.0% profit — below threshold
    target_ok  = 103.0   # 3.0% profit — above threshold
    threshold  = 0.024
    # Replicates the gate at main.py: `profit_pct = abs(target-entry)/entry; if profit_pct < 0.024: reject`
    assert abs(target_low - entry) / entry < threshold, "2% target must be rejected"
    assert abs(target_ok  - entry) / entry >= threshold, "3% target must be accepted"


def test_16_responsive_bull():
    """Three-condition fast BULL detector — all 3 must hold."""
    # Construct a clean uptrend: price rising every day. SMA50 always rising,
    # price above SMA50, recent 7 highs higher than prior 7.
    closes = [100.0 + i * 1.0 for i in range(80)]
    assert detect_responsive_bull(closes), "clean uptrend should fire responsive BULL"

    # Flat series: SMA50 not rising → must NOT fire
    flat = [100.0] * 80
    assert not detect_responsive_bull(flat), "flat series must not fire responsive BULL"

    # Downtrend → must NOT fire
    down = [200.0 - i * 1.0 for i in range(80)]
    assert not detect_responsive_bull(down), "downtrend must not fire responsive BULL"

    # Insufficient history → must NOT fire
    assert not detect_responsive_bull([100.0] * 30), "short history must not fire"

    # Rising trend BUT recent 7-day pullback breaks HH condition → must NOT fire
    closes_pullback = [100.0 + i * 1.0 for i in range(70)] + [170.0 - i for i in range(1, 8)]
    assert not detect_responsive_bull(closes_pullback), (
        "recent pullback (no HH in last 7) must not fire"
    )


def test_15_dry_run_execution():
    """Boot TravisAutoBot in DRY_RUN mode, fire a fake signal, verify no real order placed."""
    # Capture log messages emitted on the TravisAuto logger
    captured = []
    class _ListHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())
    bot_logger = logging.getLogger("TravisAuto")
    handler = _ListHandler()
    bot_logger.addHandler(handler)

    try:
        from main import TravisAutoBot
        bot = TravisAutoBot()
        assert bot.dry_run is True, "expected dry_run=True in test environment"
        before = len(bot.risk.open_trades)

        bot._execute_trade(
            pair="XBTUSD", side="buy",
            entry=65_000.0, stop=63_000.0, target=68_000.0, volume=0.01,
            confidence=0.85, regime="BULL",
        )

        after = len(bot.risk.open_trades)
        assert after == before + 1, f"expected open_trades to grow by 1 (was {before}, now {after})"

        # Verify the trade has the right shape — entry, stop, target, vol, confidence, regime
        new_trade = next(iter(bot.risk.open_trades.values()))
        assert new_trade.pair == "XBTUSD"
        assert new_trade.side == "buy"
        assert new_trade.entry_price == 65_000.0
        assert new_trade.stop_loss   == 63_000.0
        assert new_trade.take_profit == 68_000.0
        assert abs(new_trade.volume - 0.01) < 1e-9
        assert new_trade.confidence  == 0.85
        assert new_trade.regime      == "BULL"

        # Log should contain a "[DRY RUN]" line with the order parameters
        assert any("[DRY RUN]" in m for m in captured), "no [DRY RUN] log line emitted"
        assert any("XBTUSD" in m and "[DRY RUN]" in m for m in captured), (
            "[DRY RUN] log missing pair detail"
        )
    finally:
        bot_logger.removeHandler(handler)


# ── Runner ──────────────────────────────────────────────────────────────────

TESTS = [
    ("01 TRENDLINE",         test_01_trendline),
    ("02 EMA RIBBON",        test_02_ema_ribbon),
    ("03 VWAP",              test_03_vwap),
    ("04 MACD",              test_04_macd),
    ("05 ICHIMOKU",          test_05_ichimoku),
    ("06 OBV",               test_06_obv),
    ("07 RSI",               test_07_rsi),
    ("08 KELLY",             test_08_kelly),
    ("09 COMPOSITE SCORE",   test_09_composite_score),
    ("10 POSITION SIZING",   test_10_position_sizing),
    ("11 BULL REGIME PARAMS", test_11_bull_regime_params),
    ("12 DYNAMIC SPLIT 60/40", test_12_dynamic_split_bull),
    ("13 KILL SWITCH",       test_13_kill_switch),
    ("14 PROFIT THRESHOLD",  test_14_profit_threshold),
    ("15 DRY RUN EXECUTION", test_15_dry_run_execution),
    ("16 RESPONSIVE BULL DETECTOR", test_16_responsive_bull),
]


def main() -> int:
    print("=" * 60)
    print("TravisAuto — comprehensive system test suite")
    print("=" * 60)
    passed = 0
    failed = []
    for name, fn in TESTS:
        try:
            fn()
            print(f"  PASS   {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL   {name}  →  {e}")
            failed.append((name, "AssertionError", str(e)))
        except Exception as e:
            print(f"  ERROR  {name}  →  {type(e).__name__}: {e}")
            failed.append((name, type(e).__name__, str(e)))

    print("=" * 60)
    print(f"RESULT: {passed}/{len(TESTS)} passed, {len(failed)} failed")
    if failed:
        print("\nFailures:")
        for name, kind, msg in failed:
            print(f"  - {name} [{kind}]: {msg}")
    print("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
