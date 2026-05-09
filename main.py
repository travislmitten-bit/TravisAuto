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
from typing import Dict

import numpy as np

from config.config import CONFIG
from kraken.api import KrakenAPI
from strategy.trendline import generate_signals, Signal
from strategy.risk_management import RiskManager

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/travisauto.log"),
    ],
)
logger = logging.getLogger("TravisAuto")


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

    def _fetch_ohlcv(self, pair: str) -> Dict[str, np.ndarray]:
        raw = self.kraken.get_ohlcv(pair, interval=CONFIG.interval)
        raw = raw[-CONFIG.trendline.lookback_candles:]
        opens   = np.array([float(c[1]) for c in raw])
        highs   = np.array([float(c[2]) for c in raw])
        lows    = np.array([float(c[3]) for c in raw])
        closes  = np.array([float(c[4]) for c in raw])
        volumes = np.array([float(c[6]) for c in raw])
        return dict(opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes)

    def _update_balance(self):
        if self.dry_run:
            return
        try:
            if CONFIG.kraken.api_key:
                tb = self.kraken.get_trade_balance()
                self._account_balance = float(tb.get("e", self._account_balance))
        except Exception as e:
            logger.debug("Balance update skipped: %s", e)

    def _process_pair(self, pair: str):
        try:
            data = self._fetch_ohlcv(pair)
        except Exception as e:
            logger.error("OHLCV fetch failed for %s: %s", pair, e)
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
        )

        if not signals:
            return

        top = signals[0]
        logger.info(
            "Signal: %s | %s | price=%.4f | conf=%.2f | SL=%.4f | TP=%.4f",
            top.signal.value, pair, top.price,
            top.confidence, top.suggested_stop, top.suggested_target,
        )

        if top.confidence < 0.5:
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

        side = "buy" if top.signal in (Signal.BUY_BOUNCE, Signal.BUY_BREAK) else "sell"
        volume = self.risk.calculate_position_size(
            self._account_balance, top.price, top.suggested_stop, top.confidence
        )

        if volume <= 0:
            return

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

        self.risk.update_trailing_stops(prices, atr_values)

        for tid, exit_price, reason in self.risk.check_exits(prices):
            trade = self.risk.open_trades.get(tid)
            if not trade:
                continue
            closed = self.risk.close_trade(tid, exit_price, reason)
            if closed:
                logger.info("Exit %s | %s | reason=%s | pnl=%.4f",
                            tid, trade.pair, reason, closed.pnl)

    def run(self):
        self._running = True
        logger.info(
            "TravisAuto started | dry_run=%s | pairs=%s | interval=%dm",
            self.dry_run, CONFIG.pairs, CONFIG.interval,
        )
        while self._running:
            self._update_balance()
            self._manage_exits()
            for pair in CONFIG.pairs:
                self._process_pair(pair)
            logger.debug("Cycle complete. Sleeping %ds", CONFIG.poll_seconds)
            time.sleep(CONFIG.poll_seconds)

    def stop(self):
        self._running = False
        logger.info("TravisAuto stopping...")


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
