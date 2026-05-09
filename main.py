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
from strategy.trendline import generate_signals, Signal
from strategy.risk_management import RiskManager
from strategy.regime import Regime, RegimeParams, detect_regime, REGIME_PARAMS
from strategy.analytics import (
    calculate_adx, adx_scalars,
    kelly_fraction, rolling_sharpe,
    score_asset, compute_rotation_scalars,
    calculate_sma,
)

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
        self.risk = RiskManager(CONFIG.risk)
        self._running = False
        self._account_balance: float = 10_000.0

        # Staking yield tracking
        self._staking_income: float = 0.0
        self._last_staking_accrual: float = time.time()
        self._last_staking_log: float = time.time()

        # Regime cache
        self._regime: Regime = Regime.CHOPPY
        self._regime_params: RegimeParams = REGIME_PARAMS[Regime.CHOPPY]
        self._last_regime_update: float = 0.0

        # Asset rotation cache
        self._rotation_scalars: Dict[str, float] = {}
        self._last_rotation_update: float = 0.0

        # Daily Sharpe logging
        self._last_sharpe_log_day: Optional[int] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_low_liquidity_window(self) -> bool:
        """True between 11pm and 5am EST (low-volume hours)."""
        hour = datetime.now(EST).hour
        return hour >= 23 or hour < 5

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

    def _fetch_ohlcv(self, pair: str, interval: int = None) -> Dict[str, np.ndarray]:
        iv = interval or CONFIG.interval
        raw = self.kraken.get_ohlcv(pair, interval=iv)
        raw = raw[-CONFIG.trendline.lookback_candles:]
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

    def _fetch_btc_daily(self) -> list:
        """Fetch BTC daily closes for regime detection (1440m = daily)."""
        try:
            raw = self.kraken.get_ohlcv("XBTUSD", interval=1440)
            return [float(c[4]) for c in raw]
        except Exception as e:
            logger.debug("BTC daily fetch failed: %s", e)
            return []

    def _update_balance(self):
        if self.dry_run:
            return
        try:
            if CONFIG.kraken.api_key:
                tb = self.kraken.get_trade_balance()
                self._account_balance = float(tb.get("e", self._account_balance))
        except Exception as e:
            logger.debug("Balance update skipped: %s", e)

    # ── Regime cache ──────────────────────────────────────────────────────────

    def _refresh_regime(self):
        now = time.time()
        if now - self._last_regime_update < REGIME_TTL:
            return
        closes_daily = self._fetch_btc_daily()
        if closes_daily:
            self._regime, self._regime_params = detect_regime(closes_daily)
            logger.info(
                "Regime updated | %s | trail_mult=%.1f | max_risk=%.3f | "
                "min_conf=%.2f | min_touches=%d | size_scalar=%.1f",
                self._regime.value,
                self._regime_params.trail_mult,
                self._regime_params.max_risk_pct,
                self._regime_params.min_confidence,
                self._regime_params.min_touches,
                self._regime_params.size_scalar,
            )
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
        sharpe = rolling_sharpe(self.risk.daily_pnl_pcts, period=30)
        scalar = self._sharpe_scalar()
        logger.info(
            "Daily Sharpe (30d) = %.2f | size scalar = %.1fx%s",
            sharpe, scalar,
            " — REDUCED due to low Sharpe" if scalar < 1.0 else "",
        )
        self._last_sharpe_log_day = today

    # ── Signal processing ─────────────────────────────────────────────────────

    def _process_pair(self, pair: str):
        try:
            data = self._fetch_ohlcv(pair)
        except Exception as e:
            logger.error("OHLCV fetch failed for %s: %s", pair, e)
            return

        # ADX filter
        adx_val = calculate_adx(
            data["highs"].tolist(), data["lows"].tolist(), data["closes"].tolist()
        )
        skip, adx_size_scalar, adx_risk_cap = adx_scalars(adx_val)
        if skip:
            logger.debug("ADX %.1f < 20 — skipping %s", adx_val, pair)
            return

        signals = generate_signals(
            pair=pair,
            opens=data["opens"],
            highs=data["highs"],
            lows=data["lows"],
            closes=data["closes"],
            volumes=data["volumes"],
            cfg=CONFIG.trendline,
            rr_ratio=CONFIG.risk.default_rr_ratio,
            min_touches_override=self._regime_params.min_touches,
        )

        if not signals:
            return

        top = signals[0]
        logger.info(
            "Signal: %s | %s | price=%.4f | conf=%.2f | SL=%.4f | TP=%.4f | ADX=%.1f",
            top.signal.value, pair, top.price,
            top.confidence, top.suggested_stop, top.suggested_target, adx_val,
        )

        # Regime confidence gate
        if top.confidence < self._regime_params.min_confidence:
            logger.debug(
                "Signal rejected — conf %.2f below regime min %.2f | %s [%s]",
                top.confidence, self._regime_params.min_confidence,
                pair, self._regime.value,
            )
            return

        # Minimum profit threshold: target must be ≥2.4% from entry
        if top.price > 0:
            profit_pct = abs(top.suggested_target - top.price) / top.price
            if profit_pct < 0.024:
                logger.debug(
                    "Signal rejected — profit %.2f%% below 2.4%% minimum | %s",
                    profit_pct * 100, pair,
                )
                return

        if not self.risk.can_trade(self._account_balance):
            return

        if self.risk.already_in_pair(pair):
            return

        # Build effective risk %: chain all scalars
        kelly = kelly_fraction(self.risk.trade_history)
        base_risk = kelly if kelly > 0 else self._regime_params.max_risk_pct
        base_risk = min(base_risk, self._regime_params.max_risk_pct)
        if adx_risk_cap is not None:
            base_risk = min(base_risk, adx_risk_cap)

        rotation_scalar = self._rotation_scalars.get(pair, 1.0)
        time_scalar     = self._time_scalar()
        drawdown_scalar = self._drawdown_scalar()
        sharpe_scalar   = self._sharpe_scalar()

        effective_risk_pct = (
            base_risk
            * self._regime_params.size_scalar
            * adx_size_scalar
            * rotation_scalar
            * time_scalar
            * drawdown_scalar
            * sharpe_scalar
        )

        logger.debug(
            "Sizing %s | base_risk=%.4f | regime=%.1f | adx=%.1f | rot=%.1f | "
            "time=%.1f | dd=%.1f | sharpe=%.1f → eff=%.4f",
            pair, base_risk,
            self._regime_params.size_scalar, adx_size_scalar,
            rotation_scalar, time_scalar, drawdown_scalar, sharpe_scalar,
            effective_risk_pct,
        )

        volume = self.risk.calculate_position_size(
            self._account_balance, top.price, top.suggested_stop, effective_risk_pct,
        )

        if volume <= 0:
            return

        side = "buy" if top.signal in (Signal.BUY_BOUNCE, Signal.BUY_BREAK) else "sell"
        self._execute_trade(pair, side, top.price, top.suggested_stop, top.suggested_target, volume)

    def _execute_trade(self, pair, side, entry, stop, target, volume):
        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place %s market order | %s | vol=%.8f | SL=%.4f | TP=%.4f",
                side.upper(), pair, volume, stop, target,
            )
            self.risk.open_trade(pair, side, entry, stop, target, volume)
            return
        try:
            result = self.kraken.place_market_order(pair, side, volume, dry_run=False)
            logger.info("Order placed: %s", result)
        except Exception as e:
            logger.error("Order placement failed for %s: %s", pair, e)
            return
        self.risk.open_trade(pair, side, entry, stop, target, volume)

    # ── Exit management ───────────────────────────────────────────────────────

    def _manage_exits(self):
        prices: Dict[str, float] = {}
        atr_values: Dict[str, float] = {}

        for trade in self.risk.open_trades.values():
            try:
                prices[trade.pair] = self.kraken.get_mid_price(trade.pair)
                data = self._fetch_ohlcv(trade.pair)
                atr_values[trade.pair] = RiskManager.compute_atr(
                    data["highs"].tolist(), data["lows"].tolist(), data["closes"].tolist()
                )
            except Exception as e:
                logger.debug("Price fetch error for %s: %s", trade.pair, e)

        # R-milestone: 2R partial exit + breakeven SL, 3R trailing
        for tid, exit_price in self.risk.update_r_milestones(prices):
            closed_vol = self.risk.partial_close_trade(tid, exit_price)
            if closed_vol and self.dry_run:
                logger.info("[DRY RUN] Partial 50%% exit | trade %s | %.8f units @ %.4f",
                            tid, closed_vol, exit_price)

        # ATR trailing (only for pre-partial trades; pass regime trail_mult)
        self.risk.update_trailing_stops(
            prices, atr_values, trail_mult=self._regime_params.trail_mult
        )

        # Full exits: stop loss or take profit
        for tid, exit_price, reason in self.risk.check_exits(prices):
            trade = self.risk.open_trades.get(tid)
            if not trade:
                continue

            # At take-profit: do 70/30 split instead of full close
            if reason == "take_profit":
                wide_stop = self._fetch_weekly_20ma(trade.pair)
                if wide_stop > 0:
                    closed_vol = self.risk.exit_70_hold_30(tid, exit_price, wide_stop)
                    if closed_vol is not None:
                        if self.dry_run:
                            logger.info(
                                "[DRY RUN] 70%% TP exit | trade %s | %.8f @ %.4f | "
                                "holding 30%% with weekly 20MA stop %.4f",
                                tid, closed_vol, exit_price, wide_stop,
                            )
                        continue  # trade still open (30% leg)
                    # hold_30_active already set — fall through to full close
                else:
                    logger.warning("Weekly 20MA unavailable for %s — full close at TP", trade.pair)

            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info(
                    "Exit %s | %s | reason=%s | pnl=%.4f | staking_total=%.4f",
                    tid, trade.pair, reason, closed.pnl, self._staking_income,
                )

        # Hold-30 leg: exit if weekly close crossed the 20-week MA
        weekly_20ma: Dict[str, float] = {}
        for trade in self.risk.open_trades.values():
            if trade.hold_30_active and trade.pair not in weekly_20ma:
                weekly_20ma[trade.pair] = self._fetch_weekly_20ma(trade.pair)

        for tid, exit_price, reason in self.risk.check_hold_exits(weekly_20ma):
            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info(
                    "Hold-30 exit %s | %s | reason=%s | pnl=%.4f",
                    tid, closed.pair, reason, closed.pnl,
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
            self._accrue_staking()
            self.risk.tick_daily(self._account_balance)
            self._refresh_regime()
            self._refresh_rotation()
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
