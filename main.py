"""
TravisAuto — Tori's Trendline Strategy Bot
==========================================
Run:  python main.py
      DRY_RUN=false python main.py   (live trading — USE WITH CAUTION)
"""

from __future__ import annotations
import logging
import os

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

from config.config import CONFIG
from kraken.api import KrakenAPI
from strategy.trendline import generate_signals, Signal, detect_trendlines
from strategy.risk_management import RiskManager
from strategy.regime import (
    Regime, RegimeParams, detect_regime,
    detect_hh_hl_structure, detect_ma_uptrend, detect_responsive_bull,
    REGIME_PARAMS,
)
from strategy.analytics import (
    calculate_adx, adx_scalars,
    kelly_fraction, rolling_sharpe,
    score_asset, compute_rotation_scalars,
    calculate_sma,
)
from strategy.data_scanner import DataScanner, ScanResult, neutral_result
from strategy.volatility import VolatilityTracker, pearson_correlation, PAIR_TO_SYMBOL
from strategy.ema_ribbon import (
    evaluate_ribbon, RibbonResult, RibbonState, ribbon_composite_pts,
)
from strategy.indicators import (
    calculate_obv, calculate_rsi, calculate_macd, MACDResult,
    calculate_ichimoku, IchimokuResult,
    detect_hidden_divergence, detect_wyckoff_spring,
    detect_order_block, detect_fvg, calculate_fibonacci,
    OrderBlockResult, FVGResult, FibResult,
)
from notifications.telegram import TelegramNotifier

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/travisauto.log"),
    ],
)
logger = logging.getLogger("TravisAuto")

# EST = UTC-5 (no DST adjustment — conservative fixed offset)
EST = timezone(timedelta(hours=-5))
STAKING_APR       = 0.04
STAKING_LOG_INTERVAL = 3600   # log staking income once per hour
REGIME_TTL        = 3600      # re-detect regime at most once per hour
ROTATION_TTL      = 7 * 86400 # re-score assets once per week
RIBBON_TTL        = 240 * 60  # refresh ribbon once per 4h candle cycle
RIBBON_CANDLES    = 500       # need 500 4h candles for EMA(233) accuracy
BURNED_TRENDLINE_TTL = 24 * 3600   # one signal per trendline per 24h

# Per-pair min_touches overrides (LINKUSD has sparse pivot structure → 2 even in BEAR/CHOPPY)
PAIR_MIN_TOUCHES_OVERRIDE: Dict[str, int] = {
    "LINKUSD": 2,
}

# Catalyst windows — heightened caution around scheduled high-volatility events
CATALYST_EVENTS = [
    {
        "name":         "CLARITY_ACT",
        "start":        datetime(2026, 5, 13,  0,  0, tzinfo=EST),
        "end":          datetime(2026, 5, 15, 23, 59, tzinfo=EST),
        "conf_floor":   0.72,
        "size_scalar":  1.25,
        "reason":       "US crypto market-structure vote — bull capture mode (PRIMED breakout setup)",
    },
]


class TravisAutoBot:
    def __init__(self):
        self.dry_run = CONFIG.dry_run
        if not self.dry_run:
            logger.warning("LIVE TRADING MODE — real orders will be placed")

        self.kraken = KrakenAPI(
            api_key=CONFIG.kraken.api_key,
            api_secret=CONFIG.kraken.api_secret,
            base_url=CONFIG.kraken.base_url,
        )
        self.risk = RiskManager(
            CONFIG.risk,
            history_path=Path(__file__).parent / "data" / "trade_history.json",
        )
        self.scanner = DataScanner(
            taostats_api_key=CONFIG.scanner.taostats_api_key,
            lunarcrush_api_key=CONFIG.scanner.lunarcrush_api_key,
            glassnode_api_key=CONFIG.scanner.glassnode_api_key,
        )
        self._scan: ScanResult = neutral_result()
        self._vol_tracker = VolatilityTracker()   # BB compression per pair
        self._running = False
        # Trade execution is BLOCKED until the first successful Kraken balance read.
        # In dry-run mode we simulate a confirmed $100k bank so tests/sims still run.
        self._account_balance: float = 0.0
        self._balance_confirmed: bool = False
        if self.dry_run:
            self._account_balance  = 100_000.0
            self._balance_confirmed = True

        # Staking yield tracking
        self._staking_income: float = 0.0
        self._last_staking_accrual: float = time.time()
        self._last_staking_log: float = time.time()

        # Regime cache — per-asset and BTC global reference
        self._regime: Regime = Regime.CHOPPY
        self._regime_params: RegimeParams = REGIME_PARAMS[Regime.CHOPPY]
        self._asset_regimes: Dict[str, Regime] = {}
        self._asset_regime_params: Dict[str, RegimeParams] = {}
        self._last_regime_update: float = 0.0

        # Momentum breakout — pair → expiry timestamp (2 × 4H = 8h window)
        self._momentum_breakout: Dict[str, float] = {}

        # Asset rotation cache
        self._rotation_scalars: Dict[str, float] = {}
        self._last_rotation_update: float = 0.0

        # Daily Sharpe logging
        self._last_sharpe_log_day: Optional[int] = None

        # EMA ribbon cache
        self._ribbon_cache: Dict[str, RibbonResult] = {}
        self._last_ribbon_update: float = 0.0
        self._ema8_21_tightened: set = set()   # trade IDs already tightened

        # V8 Elite — per-trade exit-indicator state
        self._obv_bear_tightened: set = set()
        self._macd_contracting_active: set = set()

        # V8 Part 2 — pyramiding and BB Walk state
        self._pyramid_checked: set = set()   # (trade_id, level) pairs already attempted

        # Burned trendline registry — pair → list[(slope, intercept, expiry_ts)]
        self._burned_lines: Dict[str, list] = {}

        # Wyckoff spring 48-hour cooldown per pair
        self._wyckoff_last_fired: Dict[str, float] = {}

        # TEST_TRADE one-shot — runs once when env flag is true + balance confirmed
        self._test_trade_completed: bool = False

        # Telegram alerts (no-op if env vars missing)
        self._telegram = TelegramNotifier()
        # One-shot transition flags so we alert exactly once per state change
        self._alerted_kill_switch:    bool = False
        self._alerted_clarity_active: bool = False
        # Track per-pair regimes from prior cycle so regime flips can be alerted
        self._reconciled_on_startup: bool = False

        # Startup balance verification (live mode only) — reads free ZUSD specifically.
        # Position sizing uses CASH available, not total equity (which includes BTC etc).
        if not self.dry_run and CONFIG.kraken.api_key:
            for attempt in range(3):
                try:
                    balances = self.kraken.get_balance()
                    free_usd = float(balances.get("ZUSD", balances.get("USD", 0.0)))
                    self._account_balance  = free_usd
                    self._balance_confirmed = True
                    logger.warning(
                        "STARTUP BALANCE | free ZUSD = $%.2f | trade execution ENABLED",
                        free_usd,
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(
                            "STARTUP BALANCE | attempt %d/3 failed: %s — retrying",
                            attempt + 1, e,
                        )
                        time.sleep(2)
                    else:
                        logger.critical(
                            "STARTUP BALANCE | unable to fetch Kraken balance after 3 attempts: %s "
                            "| balance=$0.00 — TRADE EXECUTION BLOCKED until first successful read",
                            e,
                        )
        # Seed equity-curve peak only when balance is real
        if self._balance_confirmed:
            self.risk.update_peak(self._account_balance)

        # Crash/restart alert (fires every process start)
        self._telegram.send("[TravisAuto] bot started — entering main loop")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_low_liquidity_window(self) -> bool:
        """True between 11pm and 5am EST (low-volume hours)."""
        hour = datetime.now(EST).hour
        return hour >= 23 or hour < 5

    def _active_catalyst(self) -> Optional[dict]:
        """Return the active catalyst event dict if 'now' falls inside one of CATALYST_EVENTS."""
        now = datetime.now(EST)
        for evt in CATALYST_EVENTS:
            if evt["start"] <= now <= evt["end"]:
                return evt
        return None

    # ── Burned trendline registry ────────────────────────────────────────────

    def _purge_burned(self, pair: str):
        now = time.time()
        self._burned_lines[pair] = [
            (s, i, exp) for (s, i, exp) in self._burned_lines.get(pair, [])
            if exp > now
        ]

    def _is_line_burned(self, pair: str, line) -> bool:
        self._purge_burned(pair)
        ref = abs(line.intercept) + 1e-9
        for s, i, _ in self._burned_lines.get(pair, []):
            slope_denom = max(abs(line.slope), abs(s), 1e-9)
            slope_match = abs(line.slope - s) / slope_denom < 0.05
            int_match   = abs(line.intercept - i) / ref < 0.005
            if slope_match and int_match:
                return True
        return False

    def _mark_line_burned(self, pair: str, line):
        self._burned_lines.setdefault(pair, []).append(
            (line.slope, line.intercept, time.time() + BURNED_TRENDLINE_TTL)
        )

    def _time_scalar(self) -> float:
        if self._is_low_liquidity_window():
            logger.info("Low-liquidity window active (11pm–5am EST) — position sizes reduced 50%%")
            return 0.5
        return 1.0

    def _drawdown_scalar(self) -> float:
        if self.risk.drawdown_protection_active:
            return 0.5
        return 1.0

    def _sharpe_scalar(self) -> float:
        sharpe = rolling_sharpe(self.risk.daily_pnl_pcts, period=30)
        if sharpe < 1.0:
            return 0.5
        return 1.0

    # ── Scanner helpers ───────────────────────────────────────────────────────

    def _refresh_scanner(self):
        """Run the 4-hour data scan (respects internal cache TTL)."""
        try:
            self._scan = self.scanner.refresh()
        except Exception as e:
            logger.warning("DataScanner refresh failed: %s — using last result", e)

    def _effective_confidence_min(self, pair: str) -> float:
        """Asset regime min_confidence adjusted by scanner sentiment, OI, momentum, and NUPL."""
        asset_params = self._asset_regime_params.get(pair, self._regime_params)
        base  = asset_params.min_confidence
        delta = self._scan.confidence_delta
        if self._scan.oi_signal == "WEAK_MOVE":
            delta += 0.05
        if self._scan.nupl_signal == "OPPORTUNISTIC":
            delta -= 0.05
        conf_min = max(0.0, min(1.0, base + delta))
        if self._scan.nupl_signal == "CAUTION":
            conf_min = max(conf_min, 0.90)
        if time.time() < self._momentum_breakout.get(pair, 0.0):
            conf_min = min(conf_min, 0.75)
        return conf_min

    def _check_momentum_breakout(self, pair: str, data: dict) -> bool:
        """
        Flag MOMENTUM_BREAKOUT when last 4H candle moves 5%+ on above-average volume.
        Reduces confidence threshold to 0.75 for 2 × 4H scan cycles (8 hours).
        """
        closes  = data["closes"]
        volumes = data["volumes"]
        if len(closes) < 22 or len(volumes) < 22:
            return False
        move_pct  = abs(float(closes[-1]) - float(closes[-2])) / (float(closes[-2]) + 1e-9)
        avg_vol   = float(volumes[-21:-1].mean())
        vol_ratio = float(volumes[-1]) / (avg_vol + 1e-9)
        if move_pct >= 0.05 and vol_ratio > 1.0:
            already_active = time.time() < self._momentum_breakout.get(pair, 0.0)
            self._momentum_breakout[pair] = time.time() + 8 * 3600  # 2 × 4H cycles
            if not already_active:
                sign = "+" if float(closes[-1]) > float(closes[-2]) else "-"
                logger.warning(
                    "MOMENTUM_BREAKOUT | %s | move=%s%.1f%% | vol=%.1fx avg | "
                    "conf_threshold reduced to 0.75 for 2 cycles",
                    pair, sign, move_pct * 100, vol_ratio,
                )
            return True
        return False

    def _correlation_scalar(self, pair: str, new_closes: list) -> float:
        """
        Feature 4 — Correlation-adjusted position sizing.
        If the new asset's 30-day 4h price series correlates > 0.75 with any
        open position, reduce the new position to 60% of calculated size.
        """
        if not self.risk.open_trades:
            return 1.0
        for trade in self.risk.open_trades.values():
            if trade.pair == pair:
                continue
            try:
                data = self._fetch_ohlcv(trade.pair)
                corr = pearson_correlation(new_closes, data["closes"].tolist(), n=180)
                if abs(corr) > 0.75:
                    logger.warning(
                        "CORRELATION SIZING | %s↔%s r=%.2f>0.75 | "
                        "new position reduced to 60%%",
                        pair, trade.pair, corr,
                    )
                    return 0.60
            except Exception as e:
                logger.debug("Correlation check failed %s/%s: %s", pair, trade.pair, e)
        return 1.0

    # ── Staking yield ─────────────────────────────────────────────────────────

    def _accrue_staking(self):
        """Simulate 4% APR yield on idle balance when no trades are open."""
        if self.risk.open_trades:
            self._last_staking_accrual = time.time()
            return
        now = time.time()
        elapsed = now - self._last_staking_accrual
        accrued = self._account_balance * (STAKING_APR / 365 / 86400) * elapsed
        self._staking_income += accrued
        self._last_staking_accrual = now

        if now - self._last_staking_log >= STAKING_LOG_INTERVAL:
            logger.info(
                "Staking yield | accrued this cycle: $%.4f | total accumulated: $%.4f "
                "(%.4f%% APR on $%.2f idle balance)",
                accrued, self._staking_income,
                STAKING_APR * 100, self._account_balance,
            )
            self._last_staking_log = now

    # ── Data ──────────────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, pair: str, interval: int = None, limit: int = None) -> Dict[str, np.ndarray]:
        iv = interval or CONFIG.interval
        raw = self.kraken.get_ohlcv(pair, interval=iv)
        raw = raw[-(limit or CONFIG.trendline.lookback_candles):]
        opens   = np.array([float(c[1]) for c in raw])
        highs   = np.array([float(c[2]) for c in raw])
        lows    = np.array([float(c[3]) for c in raw])
        closes  = np.array([float(c[4]) for c in raw])
        volumes = np.array([float(c[6]) for c in raw])
        return dict(opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes)

    def _fetch_weekly_20ma(self, pair: str) -> float:
        """Fetch weekly candles and return the 20-week SMA of closes."""
        try:
            raw = self.kraken.get_ohlcv(pair, interval=10080)  # 10080 = 1 week
            closes = [float(c[4]) for c in raw]
            return calculate_sma(closes, 20)
        except Exception as e:
            logger.debug("Weekly 20MA fetch failed for %s: %s", pair, e)
            return 0.0

    def _fetch_asset_daily(self, pair: str) -> list:
        """Fetch daily closes for any pair for regime detection (1440m = daily)."""
        try:
            raw = self.kraken.get_ohlcv(pair, interval=1440)
            return [float(c[4]) for c in raw]
        except Exception as e:
            logger.debug("Daily fetch failed for %s: %s", pair, e)
            return []

    # ── Startup reconciliation ───────────────────────────────────────────────

    def _startup_reconciliation(self):
        """
        Fetch Kraken open orders and reconcile with any positions the bot tracks.
        On a clean restart with no persisted open trades, this logs 0/0/0. If
        we ever add open-trade persistence the same flow ensures every tracked
        position has a matching broker stop. Idempotent — runs once per startup.
        """
        if self._reconciled_on_startup:
            return
        if not self._balance_confirmed or self.dry_run:
            return
        try:
            open_orders = self.kraken.get_open_orders()
            orders = open_orders.get("open", {}) if isinstance(open_orders, dict) else {}
            stops = []
            for txid, info in orders.items():
                descr = info.get("descr", {}) or {}
                if descr.get("ordertype") in ("stop-loss", "stop-loss-limit"):
                    stops.append((txid, descr.get("pair", ""), descr.get("type", ""), info))

            n_pos      = len(self.risk.open_trades)
            n_verified = 0
            n_placed   = 0
            for tid, trade in self.risk.open_trades.items():
                want_side = self._opposite_side(trade.side)
                norm_pair = trade.pair.replace("/", "")
                match = next(
                    (s for s in stops
                     if s[1].replace("/", "") == norm_pair and s[2] == want_side),
                    None,
                )
                if match:
                    trade.broker_stop_txid = match[0]
                    n_verified += 1
                    logger.warning(
                        "STARTUP_RECONCILIATION | %s %s | existing stop verified | txid=%s",
                        tid, trade.pair, match[0],
                    )
                else:
                    new_txid = self._place_broker_stop(
                        trade.pair, want_side, trade.volume, trade.stop_loss,
                    )
                    if new_txid:
                        trade.broker_stop_txid = new_txid
                        n_placed += 1
            logger.warning(
                "STARTUP_RECONCILIATION | found %d open positions | %d stops verified | %d stops placed",
                n_pos, n_verified, n_placed,
            )
        except Exception as e:
            logger.error("STARTUP_RECONCILIATION failed: %s", e)
        finally:
            self._reconciled_on_startup = True

    # ── Kill-switch alert transition ─────────────────────────────────────────

    def _maybe_alert_kill_switch(self):
        if self.risk.kill_switch_active and not self._alerted_kill_switch:
            self._telegram.send(
                f"[TravisAuto] KILL SWITCH armed at $%.2f balance (≥20%% drawdown from peak)"
                % self._account_balance
            )
            self._alerted_kill_switch = True

    def _maybe_alert_clarity(self):
        evt = self._active_catalyst()
        if evt and not self._alerted_clarity_active:
            self._telegram.send(
                f"[TravisAuto] CATALYST WINDOW active — {evt['name']} | "
                f"conf_floor={evt['conf_floor']} size×{evt['size_scalar']}"
            )
            self._alerted_clarity_active = True
        elif not evt and self._alerted_clarity_active:
            self._alerted_clarity_active = False   # allow re-alerting on next window

    def _update_balance(self):
        if self.dry_run:
            return
        try:
            if CONFIG.kraken.api_key:
                balances = self.kraken.get_balance()
                free_usd = float(balances.get("ZUSD", balances.get("USD", 0.0)))
                self._account_balance = free_usd
                if not self._balance_confirmed:
                    self._balance_confirmed = True
                    logger.warning(
                        "BALANCE CONFIRMED | free ZUSD = $%.2f | trade execution UNBLOCKED",
                        free_usd,
                    )
                    self.risk.update_peak(free_usd)
        except Exception as e:
            logger.debug("Balance update skipped: %s", e)

    # ── TEST_TRADE one-shot ──────────────────────────────────────────────────

    def _disable_test_trade(self):
        """Atomically rewrite .env with TEST_TRADE=false and update in-memory env."""
        env_path = Path(__file__).parent / ".env"
        try:
            lines = []
            if env_path.exists():
                with env_path.open() as f:
                    lines = f.read().splitlines()
            found = False
            for i, line in enumerate(lines):
                if line.startswith("TEST_TRADE="):
                    lines[i] = "TEST_TRADE=false"
                    found = True
                    break
            if not found:
                lines.append("TEST_TRADE=false")
            tmp = env_path.with_suffix(".env.tmp")
            with tmp.open("w") as f:
                f.write("\n".join(lines) + "\n")
            tmp.replace(env_path)
            os.environ["TEST_TRADE"] = "false"
            logger.warning("TEST_TRADE | flag auto-disabled in .env (one-shot complete)")
        except Exception as e:
            logger.error("TEST_TRADE | failed to disable flag in .env: %s", e)

    def _query_order_until_filled(self, txid: str, max_wait: float = 8.0) -> dict:
        """Poll QueryOrders for status=closed (or return last snapshot on timeout)."""
        deadline = time.time() + max_wait
        last = {}
        while time.time() < deadline:
            try:
                resp = self.kraken.query_orders(txid)
                data = resp.get(txid)
                if data:
                    last = data
                    if data.get("status") == "closed":
                        return data
            except Exception as e:
                logger.debug("query_orders transient error for %s: %s", txid, e)
            time.sleep(0.5)
        return last

    def _run_test_trade(self):
        """
        One-shot live round-trip on XBTUSD: SELL 0.00015 BTC → BUY 0.00015 BTC back.
        Used when account holds BTC but needs validation that orders execute end-to-end.
        Uses self.kraken (rate-limited, signed). Auto-disables the flag on completion.
        BUY-back leg is inside try/finally so a fault on the sell-side query still
        triggers the emergency re-buy to restore the BTC position.
        """
        pair    = "XBTUSD"
        volume  = 0.00015   # ≥ Kraken $10 min notional at current BTC price
        t0 = time.time()
        logger.warning("=" * 70)
        logger.warning("TEST_TRADE | starting one-shot SELL→BUY round-trip %s @ %s BTC",
                       pair, volume)
        logger.warning("=" * 70)

        # Snapshot pre-trade balance (free ZUSD + total BTC)
        pre_zusd = None
        pre_btc  = None
        try:
            balances = self.kraken.get_balance()
            pre_zusd = float(balances.get("ZUSD", balances.get("USD", 0.0)))
            pre_btc  = float(balances.get("XXBT", balances.get("XBT",  0.0)))
            logger.warning("TEST_TRADE | pre-trade ZUSD=$%.4f | XBT=%.8f",
                           pre_zusd, pre_btc)
        except Exception as e:
            logger.error("TEST_TRADE | pre-trade balance fetch failed: %s — aborting", e)
            self._test_trade_completed = True
            self._disable_test_trade()
            return

        # Mid-price reference
        try:
            mid = self.kraken.get_mid_price(pair)
            logger.warning("TEST_TRADE | mid price = $%.2f | notional ≈ $%.4f",
                           mid, mid * volume)
        except Exception as e:
            logger.warning("TEST_TRADE | mid price fetch failed (non-fatal): %s", e)

        # ── SELL leg (opens by reducing BTC holdings, increases USD) ────────
        try:
            sell_resp = self.kraken.place_market_order(pair, "sell", volume, dry_run=False)
        except Exception as e:
            logger.critical("TEST_TRADE | SELL placement failed: %s — aborting", e)
            self._test_trade_completed = True
            self._disable_test_trade()
            return

        sell_txids = sell_resp.get("txid", [])
        if not sell_txids:
            logger.critical("TEST_TRADE | SELL returned no txid: %s — aborting", sell_resp)
            self._test_trade_completed = True
            self._disable_test_trade()
            return
        sell_txid = sell_txids[0]
        logger.warning("TEST_TRADE | SELL placed | id=%s | descr=%s",
                       sell_txid, sell_resp.get("descr", {}).get("order", "?"))

        buy_txid = None
        try:
            sell_order = self._query_order_until_filled(sell_txid)
            sell_vol      = float(sell_order.get("vol_exec", 0.0))
            sell_price    = float(sell_order.get("price",    0.0))
            sell_proceeds = float(sell_order.get("cost",     0.0))
            sell_fee      = float(sell_order.get("fee",      0.0))
            logger.warning(
                "TEST_TRADE | SELL filled | vol=%.8f @ $%.2f | proceeds=$%.4f | fee=$%.4f | status=%s",
                sell_vol, sell_price, sell_proceeds, sell_fee, sell_order.get("status"),
            )

            # ── BUY-back leg (restores BTC position) ────────────────────────
            buy_resp = self.kraken.place_market_order(pair, "buy", volume, dry_run=False)
            buy_txids = buy_resp.get("txid", [])
            if not buy_txids:
                raise RuntimeError(f"BUY-back returned no txid: {buy_resp}")
            buy_txid = buy_txids[0]
            logger.warning("TEST_TRADE | BUY-back placed | id=%s | descr=%s",
                           buy_txid, buy_resp.get("descr", {}).get("order", "?"))

            buy_order = self._query_order_until_filled(buy_txid)
            buy_vol   = float(buy_order.get("vol_exec", 0.0))
            buy_price = float(buy_order.get("price",    0.0))
            buy_cost  = float(buy_order.get("cost",     0.0))
            buy_fee   = float(buy_order.get("fee",      0.0))
            logger.warning(
                "TEST_TRADE | BUY-back filled | vol=%.8f @ $%.2f | cost=$%.4f | fee=$%.4f | status=%s",
                buy_vol, buy_price, buy_cost, buy_fee, buy_order.get("status"),
            )

            # P&L: positive if BTC dropped between SELL and BUY-back legs
            gross_pnl  = sell_proceeds - buy_cost
            total_fees = sell_fee + buy_fee
            net_pnl    = gross_pnl - total_fees
            elapsed    = time.time() - t0
            logger.warning("=" * 70)
            logger.warning("TEST_TRADE | ROUND TRIP COMPLETE (SELL→BUY)")
            logger.warning("  SELL    | %.8f @ $%.2f | proceeds=$%.4f fee=$%.4f | id=%s",
                           sell_vol, sell_price, sell_proceeds, sell_fee, sell_txid)
            logger.warning("  BUY-back| %.8f @ $%.2f | cost=$%.4f fee=$%.4f | id=%s",
                           buy_vol, buy_price, buy_cost, buy_fee, buy_txid)
            logger.warning("  Gross P&L  = $%+.4f  (positive if BTC fell between legs)",
                           gross_pnl)
            logger.warning("  Total fees = $%.4f", total_fees)
            logger.warning("  Net P&L    = $%+.4f", net_pnl)
            logger.warning("  Round-trip = %.2f s", elapsed)
            logger.warning("=" * 70)

            try:
                balances2 = self.kraken.get_balance()
                post_zusd = float(balances2.get("ZUSD", balances2.get("USD", 0.0)))
                post_btc  = float(balances2.get("XXBT", balances2.get("XBT",  0.0)))
                logger.warning(
                    "TEST_TRADE | post-trade ZUSD=$%.4f (Δ$%+.4f) | XBT=%.8f (Δ%+.8f)",
                    post_zusd, post_zusd - (pre_zusd or 0.0),
                    post_btc,  post_btc  - (pre_btc  or 0.0),
                )
            except Exception as e:
                logger.warning("TEST_TRADE | post-trade balance fetch failed: %s", e)

        except Exception as e:
            logger.critical("TEST_TRADE | round-trip error: %s", e)
            if buy_txid is None:
                logger.critical(
                    "TEST_TRADE | attempting emergency BUY-back to restore BTC position"
                )
                try:
                    er = self.kraken.place_market_order(pair, "buy", volume, dry_run=False)
                    logger.critical("TEST_TRADE | emergency buy-back submitted: %s", er)
                except Exception as ee:
                    logger.critical(
                        "TEST_TRADE | EMERGENCY BUY-BACK FAILED: %s | "
                        "MANUAL INTERVENTION REQUIRED — buy %s %s on Kraken to restore BTC",
                        ee, volume, pair,
                    )
        finally:
            self._test_trade_completed = True
            self._disable_test_trade()

    # ── Regime cache ──────────────────────────────────────────────────────────

    def _refresh_regime(self):
        now = time.time()
        # Log per-asset regimes every cycle
        if self._asset_regimes:
            logger.info(
                "ASSET_REGIME | %s",
                " | ".join(
                    f"{p}={self._asset_regimes[p].value}"
                    for p in CONFIG.pairs
                    if p in self._asset_regimes
                ),
            )
        if now - self._last_regime_update < REGIME_TTL:
            return
        for pair in CONFIG.pairs:
            # ── Manual .env override (highest priority) ──────────────────────
            env_key = f"{pair}_REGIME_OVERRIDE"
            env_val = os.environ.get(env_key, "").strip().upper()
            if env_val in (r.value for r in Regime):
                regime = Regime(env_val)
                params = REGIME_PARAMS[regime]
                self._asset_regimes[pair] = regime
                self._asset_regime_params[pair] = params
                logger.warning(
                    "REGIME OVERRIDE | %s | env %s=%s | using manual override",
                    pair, env_key, env_val,
                )
                continue

            closes_daily = self._fetch_asset_daily(pair)
            if closes_daily:
                regime, params = detect_regime(closes_daily)

                # Responsive BULL: price>SMA50 + SMA50 rising 5d + HH in last 7
                if regime != Regime.BULL:
                    try:
                        if detect_responsive_bull(closes_daily):
                            logger.warning(
                                "REGIME OVERRIDE | %s | %s → BULL | "
                                "responsive: price>SMA50 + SMA50 rising 5d + HH in 7d",
                                pair, regime.value,
                            )
                            regime = Regime.BULL
                            params = REGIME_PARAMS[Regime.BULL]
                    except Exception as e:
                        logger.debug("Responsive BULL check failed for %s: %s", pair, e)

                # 4H structural override: 10 consecutive HH+HL → BULL regardless of 200MA
                try:
                    d4h = self._fetch_ohlcv(pair)
                    if detect_hh_hl_structure(d4h["highs"].tolist(), d4h["lows"].tolist()):
                        if regime != Regime.BULL:
                            logger.warning(
                                "REGIME OVERRIDE | %s | %s → BULL | "
                                "10 consecutive 4H HH+HL (structure overrides 200MA)",
                                pair, regime.value,
                            )
                        regime = Regime.BULL
                        params = REGIME_PARAMS[Regime.BULL]
                except Exception as e:
                    logger.debug("4H HH/HL check failed for %s: %s", pair, e)

                # MA uptrend override: price > SMA50 > SMA30, weekly momentum confirms
                if regime != Regime.BULL:
                    try:
                        if detect_ma_uptrend(closes_daily):
                            logger.warning(
                                "REGIME OVERRIDE | %s | %s → BULL | "
                                "MA uptrend: price>SMA50>SMA30 + weekly momentum",
                                pair, regime.value,
                            )
                            regime = Regime.BULL
                            params = REGIME_PARAMS[Regime.BULL]
                    except Exception as e:
                        logger.debug("MA uptrend check failed for %s: %s", pair, e)

                self._asset_regimes[pair] = regime
                self._asset_regime_params[pair] = params
                logger.info(
                    "Regime updated | %s | %s | trail_mult=%.1f | max_risk=%.3f | "
                    "min_conf=%.2f | min_touches=%d | size_scalar=%.1f",
                    pair, regime.value,
                    params.trail_mult, params.max_risk_pct,
                    params.min_confidence, params.min_touches, params.size_scalar,
                )
        # Keep BTC global regime on self._regime for backward-compat (exits, logging)
        if "XBTUSD" in self._asset_regimes:
            self._regime = self._asset_regimes["XBTUSD"]
            self._regime_params = self._asset_regime_params["XBTUSD"]
        self._last_regime_update = now

    # ── Asset rotation cache ──────────────────────────────────────────────────

    def _refresh_rotation(self):
        now = time.time()
        if now - self._last_rotation_update < ROTATION_TTL:
            return
        scores: Dict[str, float] = {}
        for pair in CONFIG.pairs:
            try:
                data = self._fetch_ohlcv(pair)
                scores[pair] = score_asset(data["closes"].tolist(), data["volumes"].tolist())
            except Exception as e:
                logger.debug("Rotation score failed for %s: %s", pair, e)
                scores[pair] = 0.0
        self._rotation_scalars = compute_rotation_scalars(scores)
        logger.info("Asset rotation updated | %s", self._rotation_scalars)
        self._last_rotation_update = now

    # ── Sharpe daily log ──────────────────────────────────────────────────────

    def _log_sharpe_daily(self):
        today = datetime.now(EST).timetuple().tm_yday
        if today == self._last_sharpe_log_day:
            return
        n      = len(self.risk.daily_pnl_pcts)
        sharpe = rolling_sharpe(self.risk.daily_pnl_pcts, period=30)
        scalar = self._sharpe_scalar()
        source = "backtest baseline, no live data yet" if n < 5 else f"{n} live days"
        logger.info(
            "Daily Sharpe (30d) = %.2f [%s] | size scalar = %.1fx%s",
            sharpe, source, scalar,
            " — REDUCED due to low Sharpe" if scalar < 1.0 else "",
        )
        self._last_sharpe_log_day = today

    # ── EMA ribbon ───────────────────────────────────────────────────────────

    def _refresh_ribbon(self):
        """Pre-compute EMA ribbon for all pairs using 500 4h candles."""
        now = time.time()
        if now - self._last_ribbon_update < RIBBON_TTL:
            return
        for pair in CONFIG.pairs:
            try:
                data   = self._fetch_ohlcv(pair, limit=RIBBON_CANDLES)
                result = evaluate_ribbon(data["closes"].tolist())
                self._ribbon_cache[pair] = result
                logger.debug(
                    "Ribbon | %s | %s | strength=%.0f | composite_pts=%.1f",
                    pair, result.state.value, result.strength, result.composite_pts,
                )
            except Exception as e:
                logger.debug("Ribbon refresh failed for %s: %s", pair, e)
        self._last_ribbon_update = now

    def _ribbon_score_scalar(self) -> float:
        """Feature 3 — average ribbon composite_pts across pairs → size scalar [0.9, 1.1]."""
        if not self._ribbon_cache:
            return 1.0
        avg_pts = sum(r.composite_pts for r in self._ribbon_cache.values()) / len(self._ribbon_cache)
        return round(max(0.9, min(1.1, 1.0 + avg_pts / 100.0)), 3)

    # ── VWAP ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_daily_vwap(data: Dict[str, np.ndarray]) -> float:
        """Daily VWAP using last 6 4H candles (≈ 24h). Returns last close if volume=0."""
        h = data["highs"][-6:]
        l = data["lows"][-6:]
        c = data["closes"][-6:]
        v = data["volumes"][-6:]
        vol_sum = float(v.sum())
        if vol_sum < 1e-9:
            return float(c[-1])
        tp = (h + l + c) / 3.0
        return float((tp * v).sum() / vol_sum)

    # ── Signal processing ─────────────────────────────────────────────────────

    def _process_pair(self, pair: str):
        if not self._balance_confirmed:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=BALANCE_UNCONFIRMED | trade execution gated",
                pair,
            )
            return
        try:
            data = self._fetch_ohlcv(pair)
        except Exception as e:
            logger.error("OHLCV fetch failed for %s: %s", pair, e)
            logger.info(
                "SIGNAL_REJECTED | %s | reason=OHLCV_FETCH_FAILED | err=%s",
                pair, type(e).__name__,
            )
            return

        closes_list = data["closes"].tolist()

        # Per-asset regime params (falls back to BTC global if not yet computed)
        asset_regime = self._asset_regimes.get(pair, self._regime)
        asset_params = self._asset_regime_params.get(pair, self._regime_params)

        # Feature 1 — BB compression: update PRIMED state every cycle
        self._vol_tracker.update(pair, closes_list)

        # Momentum breakout detection: 5%+ move on above-average volume
        self._check_momentum_breakout(pair, data)

        # ADX filter
        adx_val = calculate_adx(
            data["highs"].tolist(), data["lows"].tolist(), closes_list
        )
        skip, adx_size_scalar, adx_risk_cap = adx_scalars(adx_val)
        if skip:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=ADX_TOO_LOW | adx=%.1f | threshold=20.0",
                pair, adx_val,
            )
            return
        logger.info(
            "SIGNAL_PASS | %s | filter=ADX | adx=%.1f | size_scalar=%.2f",
            pair, adx_val, adx_size_scalar,
        )

        # Per-pair override (LINKUSD's sparse structure needs touches=2)
        # takes priority over the regime's min_touches
        mt_override = PAIR_MIN_TOUCHES_OVERRIDE.get(pair, asset_params.min_touches)

        signals = generate_signals(
            pair=pair,
            opens=data["opens"],
            highs=data["highs"],
            lows=data["lows"],
            closes=data["closes"],
            volumes=data["volumes"],
            cfg=CONFIG.trendline,
            rr_ratio=CONFIG.risk.default_rr_ratio,
            min_touches_override=mt_override,
        )

        if not signals:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=NO_TRENDLINE_SIGNAL | "
                "no bounce/break at current candle (min_touches=%d)",
                pair, mt_override,
            )
            return

        top = signals[0]
        logger.info(
            "SIGNAL_PASS | %s | filter=GENERATE | n_signals=%d | top_signal=%s "
            "| top_conf=%.2f | touches=%d",
            pair, len(signals), top.signal.value, top.confidence, len(top.trendline.touches),
        )

        # Burned-trendline gate: skip if this exact line already triggered a signal in last 24h
        if self._is_line_burned(pair, top.trendline):
            logger.info(
                "SIGNAL_REJECTED | %s | reason=BURNED_TRENDLINE | "
                "this line already fired within last %dh",
                pair, BURNED_TRENDLINE_TTL // 3600,
            )
            return
        logger.info("SIGNAL_PASS | %s | filter=BURNED_LINE", pair)

        side_intent  = "buy" if top.signal in (Signal.BUY_BOUNCE, Signal.BUY_BREAK) else "sell"
        highs_list   = data["highs"].tolist()
        lows_list    = data["lows"].tolist()
        volumes_list = data["volumes"].tolist()

        # ── V8 Elite indicators (pure OHLCV — source=Kraken) ─────────────────
        obv      = calculate_obv(closes_list, volumes_list)
        rsi_vals = calculate_rsi(closes_list)
        macd_res = calculate_macd(closes_list)
        ichi     = calculate_ichimoku(highs_list, lows_list, closes_list)
        spring   = detect_wyckoff_spring(highs_list, lows_list, closes_list, volumes_list)
        obv_div  = detect_hidden_divergence(closes_list, obv)
        rsi_div  = detect_hidden_divergence(closes_list, rsi_vals)
        ob       = detect_order_block(
                       data["opens"].tolist(), highs_list, lows_list, closes_list)
        fvg      = detect_fvg(highs_list, lows_list, closes_list)
        fib      = calculate_fibonacci(highs_list, lows_list, closes_list)

        # Ichimoku hard block: price below cloud → skip all longs
        if side_intent == "buy" and ichi.below_cloud:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=PRICE_BELOW_ICHIMOKU_CLOUD | "
                "price=%.4f | cloud=[%.4f-%.4f]",
                pair, top.price, ichi.cloud_bot or 0, ichi.cloud_top or 0,
            )
            return
        logger.info(
            "SIGNAL_PASS | %s | filter=ICHIMOKU | side=%s | "
            "loc=%s | cloud_top=%.4f cloud_bot=%.4f",
            pair, side_intent,
            "ABOVE_CLOUD" if ichi.above_cloud else ("BELOW_CLOUD" if ichi.below_cloud else "IN_CLOUD"),
            ichi.cloud_top or 0, ichi.cloud_bot or 0,
        )

        # Feature 3 — LunarCrush confidence boost (+8 pts if Galaxy Score > 60)
        lc         = self._scan.lunarcrush.get(pair, {})
        lc_boost   = 0.08 if lc.get("galaxy_score", 0) > 60 else 0.0
        adj_conf   = min(1.0, top.confidence + lc_boost)
        if lc_boost:
            logger.info(
                "LunarCrush boost | %s | GS=%.0f → conf %.2f → %.2f",
                pair, lc["galaxy_score"], top.confidence, adj_conf,
            )

        # ── Feature 1 — Ribbon confirmation ──────────────────────────────────
        ribbon       = self._ribbon_cache.get(pair)
        ribbon_scalar = 1.0   # 0.75 if COMPRESSED; used in sizing below
        ribbon_boost  = 0.0

        if ribbon:
            if ribbon.state == RibbonState.BEAR_FANNING and side_intent == "buy":
                logger.info(
                    "SIGNAL_REJECTED | %s | reason=BEAR_RIBBON | "
                    "state=BEAR_FANNING | strength=%.0f | side_intent=buy",
                    pair, ribbon.strength,
                )
                return

            if ribbon.state == RibbonState.BULL_FANNING and side_intent == "buy":
                ribbon_boost = 0.08
                adj_conf     = min(1.0, adj_conf + ribbon_boost)

            if ribbon.state == RibbonState.COMPRESSED:
                ribbon_scalar = 0.75
                logger.warning(
                    "RIBBON | %s | state=COMPRESSED | strength=%.0f | "
                    "WARNING — position size reduced 25%%",
                    pair, ribbon.strength,
                )

            logger.info(
                "RIBBON | %s | state=%s | strength=%.0f | composite_pts=%.1f%s",
                pair, ribbon.state.value, ribbon.strength, ribbon.composite_pts,
                f" | confidence_boost=+{ribbon_boost:.2f}" if ribbon_boost else "",
            )

        # ── Opt 1 — BTC dominance scalar ─────────────────────────────────────
        _ALTCOINS = {"SOLUSD", "TAOUSD", "LINKUSD"}
        is_altcoin = pair in _ALTCOINS
        dom_scalar     = 1.0
        dom_conf_delta = 0.0
        if is_altcoin:
            if self._scan.dom_signal == "HEADWIND":
                dom_scalar = 0.70
                logger.info(
                    "BTC DOM HEADWIND | %s | dominance %+.2fpp in 24h | "
                    "altcoin size −30%%",
                    pair, self._scan.btc_dom_24h_change or 0,
                )
            elif self._scan.dom_signal == "TAILWIND":
                dom_conf_delta = -0.05   # lower barrier for altcoin entries
                logger.info(
                    "BTC DOM TAILWIND | %s | dominance %+.2fpp in 24h | "
                    "conf threshold −0.05",
                    pair, self._scan.btc_dom_24h_change or 0,
                )

        # ── Opt 3 — VWAP confluence ───────────────────────────────────────────
        vwap          = self._calculate_daily_vwap(data)
        price_vs_vwap = top.price - vwap
        above_vwap    = price_vs_vwap > 0
        vwap_scalar   = 1.0

        is_triple = (
            top.signal == Signal.BUY_BREAK
            and above_vwap
            and ribbon is not None
            and ribbon.state == RibbonState.BULL_FANNING
            and side_intent == "buy"
        )

        if is_triple:
            adj_conf    = min(1.0, adj_conf + 0.12)
            vwap_scalar = 1.25
            logger.info(
                "TRIPLE CONFLUENCE | %s | break+above_vwap+ribbon_bull | "
                "vwap=%.4f price=%.4f | +0.12 conf +25%% size",
                pair, vwap, top.price,
            )
        elif above_vwap and side_intent == "buy":
            adj_conf = min(1.0, adj_conf + 0.06)
            logger.info(
                "VWAP BULL CONFLUENCE | %s | price %.4f above vwap %.4f "
                "(+%.2f%%) | +0.06 conf",
                pair, top.price, vwap, price_vs_vwap / (vwap + 1e-9) * 100,
            )
        elif not above_vwap and side_intent == "buy":
            vwap_scalar = 0.80
            logger.info(
                "VWAP HEADWIND | %s | price %.4f below vwap %.4f "
                "(%.2f%%) | −20%% size",
                pair, top.price, vwap, price_vs_vwap / (vwap + 1e-9) * 100,
            )
        else:
            logger.debug(
                "VWAP | %s | price=%.4f vwap=%.4f | above=%s",
                pair, top.price, vwap, above_vwap,
            )

        # ── V8 Elite entry confirmation boosts ────────────────────────────────
        rsi_bear_scalar = 1.0
        wyckoff_scalar  = 1.0

        if side_intent == "buy":
            if obv_div.hidden_bull:
                adj_conf = min(1.0, adj_conf + 0.08)
                logger.info("OBV HIDDEN BULL | %s | price HL + OBV LL | +0.08 conf", pair)
            if rsi_div.hidden_bull:
                adj_conf = min(1.0, adj_conf + 0.06)
                logger.info("RSI HIDDEN BULL | %s | price HL + RSI LL | +0.06 conf", pair)
            if rsi_div.hidden_bear:
                rsi_bear_scalar = 0.75
                logger.warning("RSI HIDDEN BEAR | %s | price LH + RSI HH | position −25%%", pair)
            if macd_res.expanding:
                adj_conf = min(1.0, adj_conf + 0.07)
                logger.info(
                    "MACD EXPANDING | %s | hist %.4f→%.4f | +0.07 conf",
                    pair, macd_res.histogram_prev, macd_res.histogram_last,
                )
            if ichi.above_cloud:
                adj_conf = min(1.0, adj_conf + 0.06)
            if ichi.tk_cross_bull:
                adj_conf = min(1.0, adj_conf + 0.08)
                logger.info(
                    "ICHIMOKU TK CROSS | %s | Tenkan %.4f crossed above Kijun %.4f above cloud | +0.08 conf",
                    pair, ichi.tenkan or 0, ichi.kijun or 0,
                )
            cloud_state = (
                "ABOVE_CLOUD" if ichi.above_cloud
                else ("BELOW_CLOUD" if ichi.below_cloud else "IN_CLOUD")
            )
            logger.info(
                "ICHIMOKU | %s | %s | cloud[%.4f–%.4f] | T=%.4f K=%.4f | TK_cross=%s",
                pair, cloud_state,
                ichi.cloud_bot or 0, ichi.cloud_top or 0,
                ichi.tenkan or 0, ichi.kijun or 0, ichi.tk_cross_bull,
            )

            # SMC Order Block: +0.10 conf when price is back in OB range
            if ob.price_at_ob:
                adj_conf = min(1.0, adj_conf + 0.10)
                logger.info(
                    "ORDER_BLOCK | %s | price %.4f in OB [%.4f–%.4f] | +0.10 conf | source=Kraken",
                    pair, top.price, ob.ob_low, ob.ob_high,
                )

            # Fibonacci: bounce at 61.8% retracement confirmation
            if fib.at_618:
                adj_conf = min(1.0, adj_conf + 0.05)
                logger.info(
                    "FIB 61.8%% BOUNCE | %s | price %.4f ≈ fib_618 %.4f | +0.05 conf | source=Kraken",
                    pair, top.price, fib.fib_618,
                )

        logger.info(
            "MACD | %s | hist=%.4f (prev=%.4f) | expanding=%s | contracting=%s",
            pair, macd_res.histogram_last, macd_res.histogram_prev,
            macd_res.expanding, macd_res.contracting,
        )

        # Wyckoff spring: highest-conviction long — overrides all other confidence
        # 48h per-pair cooldown to prevent rapid re-fire on the same setup
        wyckoff_cooldown_until = self._wyckoff_last_fired.get(pair, 0.0) + 48 * 3600
        if spring.detected and side_intent == "buy" and time.time() >= wyckoff_cooldown_until:
            adj_conf       = 0.88
            wyckoff_scalar = 1.5
            self._wyckoff_last_fired[pair] = time.time()
            logger.warning(
                "WYCKOFF_SPRING | %s | support=%.4f | dip=%.3f%% | vol=%.1fx avg | "
                "conf=0.88 (override) | size=1.5x | next allowed in 48h",
                pair, spring.support_level, spring.penetration_pct, spring.vol_ratio,
            )
        elif spring.detected and side_intent == "buy":
            logger.debug(
                "Wyckoff suppressed for %s — within 48h cooldown (%.1fh remaining)",
                pair, max(0.0, (wyckoff_cooldown_until - time.time()) / 3600),
            )

        logger.info(
            "Signal: %s | %s | regime=%s | price=%.4f | conf=%.2f (adj=%.2f) | "
            "SL=%.4f | TP=%.4f | ADX=%.1f | primed=%s | vwap=%.4f | source=Kraken",
            top.signal.value, pair, asset_regime.value, top.price,
            top.confidence, adj_conf,
            top.suggested_stop, top.suggested_target, adx_val,
            self._vol_tracker.is_primed(pair), vwap,
        )
        if fib.swing_low > 0 and side_intent == "buy":
            logger.info(
                "FIB LEVELS | %s | swing[%.4f–%.4f] | 61.8%%=%.4f | "
                "TP targets: 127.2%%=%.4f | 161.8%%=%.4f | 261.8%%=%.4f | source=Kraken",
                pair, fib.swing_low, fib.swing_high, fib.fib_618,
                fib.fib_ext_1272, fib.fib_ext_1618, fib.fib_ext_2618,
            )
        if fvg.nearest_bullish_fvg and side_intent == "buy":
            logger.info(
                "FVG TP MAGNET | %s | nearest bullish gap [%.4f–%.4f] | source=Kraken",
                pair, fvg.nearest_bullish_fvg[0], fvg.nearest_bullish_fvg[1],
            )

        # Per-asset confidence gate (uses asset regime + momentum breakout override)
        conf_min = max(0.0, self._effective_confidence_min(pair) + dom_conf_delta)

        # Catalyst mode: MOMENTUM_BREAKOUT + BULL regime = HIGH_CONVICTION_SETUP
        momentum_active = time.time() < self._momentum_breakout.get(pair, 0.0)
        catalyst_mode   = momentum_active and asset_regime == Regime.BULL
        catalyst_scalar = 1.0
        if catalyst_mode:
            conf_min        = min(conf_min, 0.72)
            catalyst_scalar = 1.5
            logger.warning(
                "HIGH_CONVICTION_SETUP | %s | MOMENTUM_BREAKOUT + BULL regime | "
                "conf_threshold=0.72 | size=1.5x",
                pair,
            )

        # Scheduled catalyst window (e.g. CLARITY Act): raise conf floor + shrink size
        evt = self._active_catalyst()
        event_scalar = 1.0
        if evt:
            conf_min     = max(conf_min, evt["conf_floor"])
            event_scalar = evt["size_scalar"]
            logger.warning(
                "CATALYST WINDOW | %s | %s | conf_min ≥ %.2f | size ×%.2f | %s",
                pair, evt["name"], evt["conf_floor"], evt["size_scalar"], evt["reason"],
            )

        if adj_conf < conf_min:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=CONFIDENCE_TOO_LOW | "
                "adj_conf=%.2f | threshold=%.2f | regime=%s | scan=%s",
                pair, adj_conf, conf_min, asset_regime.value, self._scan.label,
            )
            return
        logger.info(
            "SIGNAL_PASS | %s | filter=CONFIDENCE | adj_conf=%.2f | threshold=%.2f",
            pair, adj_conf, conf_min,
        )

        # Minimum profit threshold: ≥2.4% from entry
        if top.price > 0:
            profit_pct = abs(top.suggested_target - top.price) / top.price
            if profit_pct < 0.024:
                logger.info(
                    "SIGNAL_REJECTED | %s | reason=PROFIT_TOO_LOW | "
                    "profit_pct=%.2f%% | threshold=2.40%% | entry=%.4f | target=%.4f",
                    pair, profit_pct * 100, top.price, top.suggested_target,
                )
                return
            logger.info(
                "SIGNAL_PASS | %s | filter=PROFIT | profit_pct=%.2f%% | target=%.4f",
                pair, profit_pct * 100, top.suggested_target,
            )

        if not self.risk.can_trade(self._account_balance):
            # risk.can_trade already logs the specific reason (kill switch / pause /
            # daily loss / max open trades). Append per-pair line for traceability.
            logger.info(
                "SIGNAL_REJECTED | %s | reason=RISK_GATE | "
                "kill_switch=%s | open_trades=%d/%d | balance=$%.2f",
                pair, self.risk.kill_switch_active,
                len(self.risk.open_trades), self.risk.cfg.max_open_trades,
                self._account_balance,
            )
            return
        logger.info("SIGNAL_PASS | %s | filter=RISK", pair)

        if self.risk.already_in_pair(pair):
            logger.info(
                "SIGNAL_REJECTED | %s | reason=ALREADY_IN_PAIR | "
                "trade already open on this pair",
                pair,
            )
            return
        logger.info("SIGNAL_PASS | %s | filter=ALREADY_IN_PAIR", pair)

        # Build effective risk %: chain all scalars
        kelly = kelly_fraction(self.risk.trade_history)
        base_risk = kelly if kelly > 0 else asset_params.max_risk_pct
        base_risk = min(base_risk, asset_params.max_risk_pct)
        if adx_risk_cap is not None:
            base_risk = min(base_risk, adx_risk_cap)

        rotation_scalar  = self._rotation_scalars.get(pair, 1.0)
        time_scalar      = self._time_scalar()
        drawdown_scalar  = self._drawdown_scalar()
        sharpe_scalar    = self._sharpe_scalar()
        scan_scalar      = self._scan.size_scalar
        # Feature 1 — BB compression: +50% if PRIMED
        primed_scalar    = self._vol_tracker.size_scalar(pair)
        # Feature 3 — Social momentum: +25% if AltRank improved 20+ in 24h
        social_scalar    = 1.25 if lc.get("social_momentum") else 1.0
        corr_scalar         = self._correlation_scalar(pair, closes_list)
        ribbon_score_scalar = self._ribbon_score_scalar()
        # L/S ratio: 1.25 if extreme shorts, 0.75 if overleveraged longs
        ls_scalar           = self._scan.ls_size_adjs.get(pair, 1.0)
        # Sector rotation from GS 24h momentum: 1.4 for rising leader, 0.7 for declining
        sector_rot_scalar   = self._scan.sector_rotation.get(pair, 1.0)

        effective_risk_pct = (
            base_risk
            * asset_params.size_scalar
            * adx_size_scalar
            * rotation_scalar
            * time_scalar
            * drawdown_scalar
            * sharpe_scalar
            * scan_scalar
            * primed_scalar
            * social_scalar
            * corr_scalar
            * ribbon_scalar
            * ribbon_score_scalar
            * dom_scalar
            * vwap_scalar
            * catalyst_scalar
            * wyckoff_scalar
            * rsi_bear_scalar
            * ls_scalar
            * sector_rot_scalar
            * event_scalar
        )

        logger.debug(
            "Sizing %s | base=%.4f | regime=%s(%.1f) | adx=%.1f | rot=%.1f | "
            "time=%.1f | dd=%.1f | sharpe=%.1f | scan=%.1f | primed=%.1f | "
            "social=%.2f | corr=%.2f | ribbon=%.2f | ribbon_score=%.3f | "
            "dom=%.2f | vwap=%.2f | catalyst=%.2f | wyckoff=%.2f | rsi_bear=%.2f | "
            "ls=%.2f | sect_rot=%.2f → eff=%.4f",
            pair, base_risk,
            asset_regime.value, asset_params.size_scalar, adx_size_scalar, rotation_scalar,
            time_scalar, drawdown_scalar, sharpe_scalar, scan_scalar, primed_scalar,
            social_scalar, corr_scalar, ribbon_scalar, ribbon_score_scalar,
            dom_scalar, vwap_scalar, catalyst_scalar, wyckoff_scalar, rsi_bear_scalar,
            ls_scalar, sector_rot_scalar, effective_risk_pct,
        )

        volume = self.risk.calculate_position_size(
            self._account_balance, top.price, top.suggested_stop, effective_risk_pct,
        )

        if volume <= 0:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=VOLUME_ZERO | "
                "effective_risk_pct=%.6f | balance=$%.2f | entry=%.4f | stop=%.4f",
                pair, effective_risk_pct, self._account_balance,
                top.price, top.suggested_stop,
            )
            return

        # ── ZUSD sufficiency gate ────────────────────────────────────────────
        # Subtract committed capital across all open positions from free ZUSD.
        # Conservative: handles the stale-balance window between cycles.
        committed_zusd = sum(
            t.entry_price * t.volume for t in self.risk.open_trades.values()
        )
        available_zusd = self._account_balance - committed_zusd
        required_zusd  = top.price * volume
        if required_zusd > available_zusd:
            logger.info(
                "SIGNAL_REJECTED | %s | reason=INSUFFICIENT_ZUSD | "
                "required=$%.4f | available=$%.4f | free=$%.4f | committed=$%.4f",
                pair, required_zusd, available_zusd,
                self._account_balance, committed_zusd,
            )
            return
        logger.info(
            "SIGNAL_PASS | %s | filter=ZUSD | required=$%.4f | available=$%.4f",
            pair, required_zusd, available_zusd,
        )

        logger.info(
            "SIGNAL_PASS | %s | filter=ALL | volume=%.8f | "
            "effective_risk_pct=%.4f%% | proceeding to execute_trade",
            pair, volume, effective_risk_pct * 100,
        )

        side = "buy" if top.signal in (Signal.BUY_BOUNCE, Signal.BUY_BREAK) else "sell"
        self._execute_trade(
            pair, side, top.price, top.suggested_stop, top.suggested_target, volume,
            confidence=adj_conf, regime=asset_regime.value,
        )
        # Feature 1 — consume PRIMED state now that signal fired
        self._vol_tracker.reset(pair)
        # Mark this trendline burned for 24h — prevents re-fire on every poll
        self._mark_line_burned(pair, top.trendline)

    @staticmethod
    def _opposite_side(side: str) -> str:
        return "sell" if side == "buy" else "buy"

    def _place_broker_stop(self, pair: str, exit_side: str,
                           volume: float, stop_price: float) -> str:
        """
        Place broker-side stop-loss (market on trigger), THEN poll once to confirm
        the order is open/pending. Returns the txid on success; "" on any failure
        (placement error, no txid, or confirmation mismatch). Caller treats "" as
        a fatal failure and must NOT leave the position unprotected.
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place BROKER STOP | %s %s | vol=%.8f stop=%.4f",
                pair, exit_side.upper(), volume, stop_price,
            )
            return "DRYRUN-STOP"
        try:
            resp = self.kraken.place_stop_loss_market(
                pair, exit_side, volume, stop_price, dry_run=False,
            )
            txid = (resp.get("txid") or [""])[0]
            if not txid:
                logger.critical(
                    "STOP_PLACEMENT_FAILED | %s | no txid returned: %s", pair, resp,
                )
                return ""
        except Exception as e:
            logger.critical(
                "STOP_PLACEMENT_FAILED | %s | %.8f %s @ stop=%.4f | placement err=%s",
                pair, volume, exit_side.upper(), stop_price, e,
            )
            return ""

        # Confirm the stop is actually live on the book.
        time.sleep(0.5)
        try:
            info = self.kraken.query_orders(txid)
            data = info.get(txid, {}) if isinstance(info, dict) else {}
            status = (data.get("status") or "").lower()
            if status not in ("open", "pending"):
                logger.critical(
                    "STOP_PLACEMENT_FAILED | %s | txid=%s status=%s | not active",
                    pair, txid, status,
                )
                # Best-effort cancel of the unconfirmed order so it can't fire later.
                try:
                    self.kraken.cancel_order(txid)
                except Exception:
                    pass
                return ""
            logger.warning(
                "BROKER STOP CONFIRMED | %s | %.8f %s @ stop=%.4f | txid=%s | status=%s",
                pair, volume, exit_side.upper(), stop_price, txid, status,
            )
            return txid
        except Exception as e:
            logger.critical(
                "STOP_PLACEMENT_FAILED | %s | txid=%s | confirmation query failed: %s",
                pair, txid, e,
            )
            return ""

    def _cancel_broker_stop(self, trade) -> bool:
        """Cancel an existing broker stop. Best-effort; doesn't raise."""
        if not trade.broker_stop_txid or trade.broker_stop_txid == "DRYRUN-STOP":
            return True
        if self.dry_run:
            return True
        try:
            self.kraken.cancel_order(trade.broker_stop_txid)
            logger.info("Broker stop canceled | %s | txid=%s",
                        trade.id, trade.broker_stop_txid)
            trade.broker_stop_txid = ""
            return True
        except Exception as e:
            logger.warning(
                "Broker stop cancel failed (proceeding anyway) | %s | txid=%s | err=%s",
                trade.id, trade.broker_stop_txid, e,
            )
            return False

    def _broker_close_position(self, trade, volume: float, reason: str) -> bool:
        """
        Cancel broker stop + place a market order to close `volume`. Returns True on
        broker-side success. On failure, no in-memory state should be mutated by the
        caller — the position remains tracked and the bot will retry next cycle.
        """
        exit_side = self._opposite_side(trade.side)
        if self.dry_run:
            logger.info(
                "[DRY RUN] EXIT ORDER | %s %s | %.8f %s | reason=%s",
                trade.id, trade.pair, volume, exit_side.upper(), reason,
            )
            self._cancel_broker_stop(trade)
            return True
        # Cancel the existing broker stop FIRST so it can't double-fill against our market sell.
        self._cancel_broker_stop(trade)
        try:
            result = self.kraken.place_market_order(
                trade.pair, exit_side, volume, dry_run=False,
            )
            txid = (result.get("txid") or [""])[0]
            logger.warning(
                "EXIT ORDER PLACED | %s %s | %.8f %s | reason=%s | txid=%s",
                trade.id, trade.pair, volume, exit_side.upper(), reason, txid,
            )
            return True
        except Exception as e:
            logger.critical(
                "EXIT ORDER FAILED | %s %s | %.8f %s | reason=%s | err=%s | "
                "position still open on exchange — will retry next cycle",
                trade.id, trade.pair, volume, exit_side.upper(), reason, e,
            )
            return False

    def _replace_broker_stop(self, trade, new_volume: float, new_stop_price: float):
        """After a partial exit, install a fresh broker stop for the held portion."""
        exit_side = self._opposite_side(trade.side)
        new_txid = self._place_broker_stop(trade.pair, exit_side, new_volume, new_stop_price)
        trade.broker_stop_txid = new_txid

    def _execute_trade(self, pair, side, entry, stop, target, volume,
                       confidence: float = 0.0, regime: str = ""):
        if not self._balance_confirmed:
            logger.warning(
                "TRADE BLOCKED | %s %s — balance not yet confirmed by Kraken",
                side.upper(), pair,
            )
            return
        asset_params = self._asset_regime_params.get(pair, self._regime_params)
        asset_regime = self._asset_regimes.get(pair, self._regime)
        hold_pct = asset_params.hold_pct
        exit_pct = round(1.0 - hold_pct, 2)
        regime_name = regime or asset_regime.value
        exit_side  = self._opposite_side(side)
        entry_txid = ""
        broker_stop_txid = ""

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place %s market order | %s | vol=%.8f | SL=%.4f | TP=%.4f "
                "| TP split %.0f%%/%.0f%% exit/hold [%s regime]",
                side.upper(), pair, volume, stop, target,
                exit_pct * 100, hold_pct * 100, regime_name,
            )
            broker_stop_txid = self._place_broker_stop(pair, exit_side, volume, stop)
            self.risk.open_trade(pair, side, entry, stop, target, volume,
                                 confidence=confidence, regime=regime_name,
                                 entry_txid="DRYRUN-ENTRY",
                                 broker_stop_txid=broker_stop_txid)
            return
        try:
            result = self.kraken.place_market_order(pair, side, volume, dry_run=False)
            entry_txid = (result.get("txid") or [""])[0]
            logger.warning(
                "ENTRY ORDER PLACED | %s %s | vol=%.8f | txid=%s | "
                "TP split %.0f%%/%.0f%% exit/hold [%s regime]",
                pair, side.upper(), volume, entry_txid,
                exit_pct * 100, hold_pct * 100, regime_name,
            )
        except Exception as e:
            logger.error("Order placement failed for %s: %s", pair, e)
            return
        # Place exchange-side hard stop immediately after entry fills.
        # If we cannot CONFIRM the stop is live, we IMMEDIATELY close the entry
        # to avoid holding an unprotected position. _place_broker_stop returns ""
        # for any failure (placement, missing txid, status not open/pending).
        broker_stop_txid = self._place_broker_stop(pair, exit_side, volume, stop)
        if not broker_stop_txid:
            logger.critical(
                "UNPROTECTED ENTRY | %s | stop placement could not be confirmed — "
                "closing entry immediately to avoid naked position",
                pair,
            )
            try:
                close_resp = self.kraken.place_market_order(
                    pair, exit_side, volume, dry_run=False,
                )
                close_txid = (close_resp.get("txid") or [""])[0]
                logger.critical(
                    "UNPROTECTED ENTRY closed | %s | emergency %s txid=%s",
                    pair, exit_side.upper(), close_txid,
                )
                self._telegram.send(
                    f"[TravisAuto] UNPROTECTED ENTRY auto-closed: {pair} "
                    f"stop placement failed, entry rolled back"
                )
            except Exception as e:
                logger.critical(
                    "EMERGENCY CLOSE FAILED | %s | err=%s | "
                    "MANUAL INTERVENTION REQUIRED — close %.8f %s on Kraken",
                    pair, e, volume, pair,
                )
                self._telegram.send(
                    f"[TravisAuto] CRITICAL: {pair} entry is naked and emergency "
                    f"close failed ({e}). Manual close required."
                )
            return
        self.risk.open_trade(pair, side, entry, stop, target, volume,
                             confidence=confidence, regime=regime_name,
                             entry_txid=entry_txid,
                             broker_stop_txid=broker_stop_txid)
        self._telegram.send(
            f"[TravisAuto] OPENED {side.upper()} {pair} vol={volume:.8f} "
            f"entry=${entry:.2f} stop=${stop:.2f} target=${target:.2f}"
        )

    # ── Multi-timeframe dynamic exit ──────────────────────────────────────────

    def _upgrade_exit_modes(
        self, prices: Dict[str, float], data_4h: Dict[str, dict]
    ):
        """
        Per cycle: evaluate each open trade for parabolic move conditions and
        upgrade (or revert) its exit_mode.

        Upgrade rules (in priority order):
          ≥15% directional move in 48h → switch to WEEKLY trendline exit
          ≥5% in 24h with no adverse 4h candle → switch to DAILY trendline exit

        Revert rules:
          WEEKLY → stays weekly until weekly trendline break or SL (no auto-revert)
          DAILY  → reverts to 4H when the daily candle closes below the daily
                   support trendline (handled in _check_elevated_exits)
        """
        for tid, trade in list(self.risk.open_trades.items()):
            pair = trade.pair
            price = prices.get(pair)
            d = data_4h.get(pair)
            if price is None or d is None:
                continue

            closes = d["closes"]
            opens  = d["opens"]
            sign   = 1 if trade.side == "buy" else -1

            move_24h = sign * (price - closes[-7])  / (closes[-7]  + 1e-9) if len(closes) >= 7  else 0.0
            move_48h = sign * (price - closes[-13]) / (closes[-13] + 1e-9) if len(closes) >= 13 else 0.0

            # No adverse candle in last 6 4h periods (= 24h)
            if len(closes) >= 6:
                if trade.side == "buy":
                    no_adverse = all(closes[i] >= opens[i] for i in range(-6, 0))
                else:
                    no_adverse = all(closes[i] <= opens[i] for i in range(-6, 0))
            else:
                no_adverse = False

            # ── Weekly upgrade ────────────────────────────────────────────────
            if move_48h >= 0.15 and trade.exit_mode != "weekly":
                old = trade.exit_mode.upper()
                trade.exit_mode = "weekly"
                trade.exit_mode_reason = (
                    f"{move_48h:.1%} directional move in 48h"
                )
                logger.warning(
                    "EXIT MODE | %s %s | %s → WEEKLY | reason: %s",
                    tid, pair, old, trade.exit_mode_reason,
                )

            # ── Daily upgrade (only from 4h, not from weekly) ─────────────────
            elif move_24h >= 0.05 and no_adverse and trade.exit_mode == "4h":
                trade.exit_mode = "daily"
                trade.exit_mode_reason = (
                    f"{move_24h:.1%} in 24h, no adverse 4h candle"
                )
                logger.warning(
                    "EXIT MODE | %s %s | 4H → DAILY | reason: %s",
                    tid, pair, trade.exit_mode_reason,
                )

    def _check_elevated_exits(
        self, prices: Dict[str, float]
    ):
        """
        For DAILY mode trades: detect if the daily candle closed below the
        daily support trendline → revert to 4h (don't close here; let normal
        SL/trail catch the exit on the next 4h candle).

        For WEEKLY mode trades: detect if the weekly candle closed below the
        weekly support trendline → close the trade immediately.

        Returns list of (trade_id, exit_price, reason) to close.
        """
        exits_to_close = []

        for tid, trade in list(self.risk.open_trades.items()):
            if not trade.exit_mode_elevated:
                continue

            pair  = trade.pair
            price = prices.get(pair, 0.0)

            if trade.exit_mode == "daily":
                try:
                    d = self._fetch_ohlcv(pair, interval=1440)
                    n = len(d["closes"])
                    sup_lines, res_lines = detect_trendlines(
                        d["opens"], d["highs"], d["lows"], d["closes"],
                        CONFIG.trendline,
                    )
                    lines = sup_lines if trade.side == "buy" else res_lines
                    if not lines:
                        continue
                    line_price = lines[0].price_at(n - 1)
                    daily_close = float(d["closes"][-1])

                    broken = (trade.side == "buy"  and daily_close < line_price) or \
                             (trade.side == "sell" and daily_close > line_price)
                    if broken:
                        logger.warning(
                            "EXIT MODE | %s %s | DAILY → 4H (revert) | "
                            "daily close %.4f crossed %s trendline %.4f — "
                            "parabolic over, reverting to 4h exit logic",
                            tid, pair, daily_close,
                            "below support" if trade.side == "buy" else "above resistance",
                            line_price,
                        )
                        trade.exit_mode = "4h"
                        trade.exit_mode_reason = (
                            f"daily close {daily_close:.4f} crossed trendline {line_price:.4f}"
                        )
                except Exception as e:
                    logger.debug("Daily trendline check failed for %s: %s", pair, e)

            elif trade.exit_mode == "weekly":
                try:
                    d = self._fetch_ohlcv(pair, interval=10080)
                    n = len(d["closes"])
                    sup_lines, res_lines = detect_trendlines(
                        d["opens"], d["highs"], d["lows"], d["closes"],
                        CONFIG.trendline,
                    )
                    lines = sup_lines if trade.side == "buy" else res_lines
                    if not lines:
                        continue
                    line_price = lines[0].price_at(n - 1)
                    weekly_close = float(d["closes"][-1])

                    broken = (trade.side == "buy"  and weekly_close < line_price) or \
                             (trade.side == "sell" and weekly_close > line_price)
                    if broken:
                        logger.warning(
                            "EXIT MODE | %s %s | WEEKLY trendline break → CLOSING | "
                            "weekly close %.4f crossed %s trendline %.4f",
                            tid, pair, weekly_close,
                            "below support" if trade.side == "buy" else "above resistance",
                            line_price,
                        )
                        exits_to_close.append((tid, price, "weekly_trendline_break"))
                except Exception as e:
                    logger.debug("Weekly trendline check failed for %s: %s", pair, e)

        return exits_to_close

    # ── Pyramiding ────────────────────────────────────────────────────────────

    def _check_pyramids(
        self,
        prices:    Dict[str, float],
        data_4h:   Dict[str, dict],
        atr_values: Dict[str, float],
    ):
        """
        Pyramid winners in BULL regime when MACD is expanding.
        2R → add 40% original position, stop at breakeven.
        4R → add 20% original position, stop at current trailing.
        Max 2 pyramids per trade.
        """
        for tid, trade in list(self.risk.open_trades.items()):
            if trade.pyramid_count >= 2 or trade.hold_30_active:
                continue
            asset_regime = self._asset_regimes.get(trade.pair, self._regime)
            if asset_regime != Regime.BULL:
                continue
            price = prices.get(trade.pair)
            d     = data_4h.get(trade.pair)
            if not price or not d:
                continue
            macd_res = calculate_macd(d["closes"].tolist())
            if not macd_res.expanding:
                continue
            sign     = 1 if trade.side == "buy" else -1
            profit_r = sign * (price - trade.entry_price) / max(trade.r_value, 1e-9)
            orig_vol = trade.original_volume if trade.original_volume > 0 else trade.volume

            if profit_r >= 2.0 and trade.pyramid_count == 0 and (tid, 2) not in self._pyramid_checked:
                self._pyramid_checked.add((tid, 2))
                add_vol  = orig_vol * 0.40
                new_stop = trade.entry_price
                if add_vol > 0:
                    self.risk.add_pyramid(tid, price, add_vol, new_stop)
                    logger.warning(
                        "PYRAMID_ENTRY | %s %s | 2R + BULL + MACD expanding | "
                        "+40%% (%.8f) @ %.4f | stop → breakeven %.4f | source=Kraken",
                        tid, trade.pair, add_vol, price, new_stop,
                    )

            elif profit_r >= 4.0 and trade.pyramid_count == 1 and (tid, 4) not in self._pyramid_checked:
                self._pyramid_checked.add((tid, 4))
                add_vol  = orig_vol * 0.20
                new_stop = trade.trailing_stop
                if add_vol > 0:
                    self.risk.add_pyramid(tid, price, add_vol, new_stop)
                    logger.warning(
                        "PYRAMID_ENTRY | %s %s | 4R + BULL + MACD expanding | "
                        "+20%% (%.8f) @ %.4f | stop → trail %.4f | source=Kraken",
                        tid, trade.pair, add_vol, price, new_stop,
                    )

    # ── Exit management ───────────────────────────────────────────────────────

    def _manage_exits(self):
        prices: Dict[str, float] = {}
        atr_values: Dict[str, float] = {}
        data_4h: Dict[str, dict] = {}

        for trade in self.risk.open_trades.values():
            try:
                prices[trade.pair] = self.kraken.get_mid_price(trade.pair)
                data = self._fetch_ohlcv(trade.pair)
                data_4h[trade.pair] = data
                atr_values[trade.pair] = RiskManager.compute_atr(
                    data["highs"].tolist(), data["lows"].tolist(), data["closes"].tolist()
                )
            except Exception as e:
                logger.debug("Price fetch error for %s: %s", trade.pair, e)

        # Dynamic exit mode: upgrade to daily/weekly on parabolic moves
        self._upgrade_exit_modes(prices, data_4h)

        # Elevated exit checks: daily revert or weekly trendline close
        for tid, exit_price, reason in self._check_elevated_exits(prices):
            trade = self.risk.open_trades.get(tid)
            if not trade:
                continue
            # Place broker close FIRST; only update internal state on broker success.
            if not self._broker_close_position(trade, trade.volume, reason):
                continue
            self._ema8_21_tightened.discard(tid)
            self._obv_bear_tightened.discard(tid)
            self._macd_contracting_active.discard(tid)
            self._pyramid_checked.discard((tid, 2))
            self._pyramid_checked.discard((tid, 4))
            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info(
                    "Exit %s | %s | reason=%s | exit_mode=weekly | pnl=%.4f",
                    tid, trade.pair, reason, closed.pnl,
                )
                self._telegram.send(
                    f"[TravisAuto] CLOSED {trade.pair} exit=${exit_price:.2f} "
                    f"P&L=${closed.pnl:+.4f} reason={reason}"
                )

        # R-milestone: 2R partial exit + breakeven SL, 3R trailing
        for tid, exit_price in self.risk.update_r_milestones(prices):
            trade = self.risk.open_trades.get(tid)
            if not trade or trade.partial_exit_done:
                continue
            half = round(trade.volume / 2, 8)
            # Broker-side: place real sell for the 50% chunk
            if not self._broker_close_position(trade, half, "partial_2R"):
                continue
            closed_vol = self.risk.partial_close_trade(tid, exit_price)
            if closed_vol is None:
                continue
            # Held leg now has 50% volume + breakeven SL — install fresh broker stop.
            self._replace_broker_stop(trade, trade.volume, trade.entry_price)
            if self.dry_run:
                logger.info("[DRY RUN] Partial 50%% exit | trade %s | %.8f units @ %.4f",
                            tid, closed_vol, exit_price)

        # ATR trailing — per-asset regime trail_mult; various conditions modify it
        trail_mults: Dict[str, float] = {
            p: self._asset_regime_params.get(p, self._regime_params).trail_mult
            for p in CONFIG.pairs
        }
        if self._scan.smart_money_divergence:
            logger.warning("SMART MONEY DIVERGENCE active | trail_mult capped to 0.5 for all pairs")
            trail_mults = {p: min(v, 0.5) for p, v in trail_mults.items()}
        # NUPL CAUTION: tighten all trailing stops to 0.5×
        if self._scan.nupl_signal == "CAUTION":
            trail_mults = {p: v * 0.5 for p, v in trail_mults.items()}
        # L/S > 2.0 (overleveraged longs): tighten stops 25% per pair
        for p in CONFIG.pairs:
            if self._scan.ls_size_adjs.get(p, 1.0) < 1.0:
                trail_mults[p] = trail_mults[p] * 0.75
        # BB Walk: 3+ consecutive closes above upper BB → weekly exit + 2× trail
        for trade in self.risk.open_trades.values():
            d = data_4h.get(trade.pair)
            if d is None:
                continue
            if self._vol_tracker.check_bb_walk(trade.pair, d["closes"].tolist()):
                trail_mults[trade.pair] = trail_mults.get(trade.pair, 1.0) * 2.0
                if trade.exit_mode != "weekly":
                    trade.exit_mode = "weekly"
                    trade.exit_mode_reason = "BB_WALK: 3+ consecutive closes above upper BB"
                    logger.warning(
                        "BB_WALK | %s %s | exit mode → WEEKLY | trail 2× normal",
                        trade.id, trade.pair,
                    )
        self.risk.update_trailing_stops(prices, atr_values, trail_mult=trail_mults)

        # ETF consecutive negative flows: tighten stops 20%
        if self._scan.etf_flow_streak <= -3:
            for tid, trade in list(self.risk.open_trades.items()):
                atr = atr_values.get(trade.pair)
                p   = prices.get(trade.pair)
                if atr and p:
                    self.risk.tighten_trailing_stop(tid, p, atr, tight_mult=0.8)

        # V8 — OBV hidden bear (one-time) + MACD contracting (ongoing) → tighten stops
        for tid, trade in list(self.risk.open_trades.items()):
            d     = data_4h.get(trade.pair)
            price = prices.get(trade.pair)
            atr   = atr_values.get(trade.pair)
            if not d or not price or not atr:
                continue
            c = d["closes"].tolist()
            v = d["volumes"].tolist()

            # OBV hidden bearish: stop tightened once per trade
            if tid not in self._obv_bear_tightened and trade.side == "buy":
                try:
                    obv_d = detect_hidden_divergence(c, calculate_obv(c, v))
                    if obv_d.hidden_bear:
                        self.risk.tighten_trailing_stop(tid, price, atr, tight_mult=0.7)
                        self._obv_bear_tightened.add(tid)
                        logger.warning(
                            "OBV HIDDEN BEAR | %s %s | price making lower highs, OBV rising | "
                            "trailing stop tightened 30%%",
                            tid, trade.pair,
                        )
                except Exception as e:
                    logger.debug("OBV bear check failed %s: %s", tid, e)

            # MACD contracting: tighten every cycle it's contracting; log on transition
            try:
                macd_exit = calculate_macd(c)
                if macd_exit.contracting and trade.side == "buy":
                    self.risk.tighten_trailing_stop(tid, price, atr, tight_mult=0.7)
                    if tid not in self._macd_contracting_active:
                        self._macd_contracting_active.add(tid)
                        logger.warning(
                            "MACD CONTRACTING | %s %s | hist %.4f→%.4f | "
                            "trailing stop tightened to 0.7×",
                            tid, trade.pair,
                            macd_exit.histogram_prev, macd_exit.histogram_last,
                        )
                elif tid in self._macd_contracting_active:
                    self._macd_contracting_active.discard(tid)
                    logger.info("MACD EXPANDING | %s %s | trailing resumed at normal rate",
                                tid, trade.pair)
            except Exception as e:
                logger.debug("MACD exit check failed %s: %s", tid, e)

        # Feature 2 — Ribbon trailing exit: 8-EMA crosses below 21-EMA → tighten to 0.5×
        for tid, trade in list(self.risk.open_trades.items()):
            if tid in self._ema8_21_tightened:
                continue
            ribbon = self._ribbon_cache.get(trade.pair)
            price  = prices.get(trade.pair)
            atr    = atr_values.get(trade.pair)
            if not ribbon or not price or not atr:
                continue
            ema8  = ribbon.emas.get(8, 0.0)
            ema21 = ribbon.emas.get(21, 0.0)
            crossed = (trade.side == "buy" and ema8 < ema21) or \
                      (trade.side == "sell" and ema8 > ema21)
            if crossed:
                self.risk.tighten_trailing_stop(tid, price, atr, tight_mult=0.5)
                self._ema8_21_tightened.add(tid)
                logger.warning(
                    "RIBBON TRAIL | %s %s | 8-EMA %.4f %s 21-EMA %.4f | "
                    "trailing stop tightened to 0.5× ATR",
                    tid, trade.pair, ema8,
                    "below" if trade.side == "buy" else "above",
                    ema21,
                )

        # Full exits: stop loss or take profit
        # Note: check_exits already suppresses TP for elevated-mode trades
        for tid, exit_price, reason in self.risk.check_exits(prices):
            trade = self.risk.open_trades.get(tid)
            if not trade:
                continue

            # At take-profit: per-asset regime-driven partial exit + hold remainder
            if reason == "take_profit":
                wide_stop = self._fetch_weekly_20ma(trade.pair)
                if wide_stop > 0:
                    _ap = self._asset_regime_params.get(trade.pair, self._regime_params)
                    hold_pct  = 0.50 if trade.exit_mode == "weekly" else _ap.hold_pct
                    exit_pct  = 1.0 - hold_pct
                    sell_vol  = round(trade.volume * exit_pct, 8)
                    # Broker-side: place real sell for the exit portion
                    if not self._broker_close_position(trade, sell_vol, "partial_tp"):
                        continue
                    closed_vol = self.risk.exit_70_hold_30(
                        tid, exit_price, wide_stop, hold_pct=hold_pct
                    )
                    if closed_vol is not None:
                        # Held leg has new volume + wide stop — install fresh broker stop.
                        self._replace_broker_stop(trade, trade.volume, wide_stop)
                        if self.dry_run:
                            logger.info(
                                "[DRY RUN] %.0f%%/%.0f%% TP exit | trade %s | "
                                "%.8f @ %.4f | holding %.0f%% with weekly 20MA stop %.4f",
                                exit_pct * 100, hold_pct * 100,
                                tid, closed_vol, exit_price, hold_pct * 100, wide_stop,
                            )
                        continue  # trade still open (hold leg)
                    # hold_30_active already set — fall through to full close
                else:
                    logger.warning("Weekly 20MA unavailable for %s — full close at TP", trade.pair)

            # Full close path (stop loss or unguarded take profit)
            if not self._broker_close_position(trade, trade.volume, reason):
                continue
            self._ema8_21_tightened.discard(tid)
            self._obv_bear_tightened.discard(tid)
            self._macd_contracting_active.discard(tid)
            self._pyramid_checked.discard((tid, 2))
            self._pyramid_checked.discard((tid, 4))
            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info(
                    "Exit %s | %s | reason=%s | exit_mode=%s | pnl=%.4f | staking_total=%.4f",
                    tid, trade.pair, reason, trade.exit_mode, closed.pnl, self._staking_income,
                )
                self._telegram.send(
                    f"[TravisAuto] CLOSED {trade.pair} exit=${exit_price:.2f} "
                    f"P&L=${closed.pnl:+.4f} reason={reason}"
                )

        # Pyramid winners: 2R/4R additions in BULL regime with MACD expanding
        self._check_pyramids(prices, data_4h, atr_values)

        # Hold-30 leg: exit if weekly close crossed the 20-week MA
        weekly_20ma: Dict[str, float] = {}
        for trade in self.risk.open_trades.values():
            if trade.hold_30_active and trade.pair not in weekly_20ma:
                weekly_20ma[trade.pair] = self._fetch_weekly_20ma(trade.pair)

        for tid, exit_price, reason in self.risk.check_hold_exits(weekly_20ma):
            trade = self.risk.open_trades.get(tid)
            if not trade:
                continue
            if not self._broker_close_position(trade, trade.volume, reason):
                continue
            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info(
                    "Hold-30 exit %s | %s | reason=%s | pnl=%.4f",
                    tid, closed.pair, reason, closed.pnl,
                )
                self._telegram.send(
                    f"[TravisAuto] CLOSED {closed.pair} (hold-30 leg) exit=${exit_price:.2f} "
                    f"P&L=${closed.pnl:+.4f} reason={reason}"
                )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        logger.info(
            "TravisAuto started | dry_run=%s | pairs=%s | interval=%dm",
            self.dry_run, CONFIG.pairs, CONFIG.interval,
        )
        while self._running:
            self._update_balance()
            if self._balance_confirmed:
                self.risk.update_peak(self._account_balance)
                self.risk.tick_daily(self._account_balance)
                # Run startup reconciliation once balance is live
                self._startup_reconciliation()
                # One-shot TEST_TRADE: fires once when env flag set + live mode
                if (
                    not self._test_trade_completed
                    and not self.dry_run
                    and os.environ.get("TEST_TRADE", "").strip().lower() == "true"
                ):
                    self._run_test_trade()
                # Periodic alert transition checks (idempotent)
                self._maybe_alert_kill_switch()
                self._maybe_alert_clarity()
            self._accrue_staking()
            self._refresh_regime()
            self._refresh_rotation()
            self._refresh_scanner()
            self._refresh_ribbon()
            self._log_sharpe_daily()
            self._manage_exits()
            for pair in CONFIG.pairs:
                self._process_pair(pair)
            logger.debug("Cycle complete. Sleeping %ds", CONFIG.poll_seconds)
            time.sleep(CONFIG.poll_seconds)

    def stop(self):
        self._running = False
        logger.info(
            "TravisAuto stopping | total staking income: $%.4f",
            self._staking_income,
        )
        self._telegram.send(
            f"[TravisAuto] bot stopping | open_trades={len(self.risk.open_trades)} "
            f"| staking=${self._staking_income:.4f}"
        )


def main():
    bot = TravisAutoBot()

    def _sig_handler(sig, frame):
        bot.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    bot.run()
    logger.info("TravisAuto shut down cleanly.")


if __name__ == "__main__":
    main()
