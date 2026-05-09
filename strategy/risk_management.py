"""
Position sizing and trade lifecycle management.
Uses fixed fractional risk: risk 1% of account per trade by default.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from config.config import RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    id: str
    pair: str
    side: str               # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    status: str = "open"    # open | closed | cancelled
    exit_price: float = 0.0
    pnl: float = 0.0
    trailing_stop: float = 0.0
    notes: str = ""
    r_value: float = 0.0           # |entry - stop|, risk per unit
    partial_exit_done: bool = False # 50% closed at 2R
    breakeven_set: bool = False     # SL moved to entry at 2R


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._open_trades: Dict[str, Trade] = {}
        self._daily_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._trade_counter: int = 0
        self._consecutive_losses: int = 0
        self._drawdown_protection: bool = False

    # ── Daily drawdown reset ──────────────────────────────────────────────────

    def _check_date_rollover(self):
        today = date.today()
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_date = today

    # ── Checks ────────────────────────────────────────────────────────────────

    def can_trade(self, account_balance: float) -> bool:
        self._check_date_rollover()
        if len(self._open_trades) >= self.cfg.max_open_trades:
            logger.warning("Max open trades reached (%d)", self.cfg.max_open_trades)
            return False
        daily_loss_pct = abs(min(0.0, self._daily_pnl)) / (account_balance + 1e-9)
        if daily_loss_pct >= self.cfg.max_daily_loss:
            logger.warning("Daily loss limit hit (%.2f%%)", daily_loss_pct * 100)
            return False
        return True

    def already_in_pair(self, pair: str) -> bool:
        return any(t.pair == pair for t in self._open_trades.values())

    # ── Drawdown protection ───────────────────────────────────────────────────

    @property
    def drawdown_protection_active(self) -> bool:
        return self._drawdown_protection

    def record_trade_result(self, won: bool):
        if won:
            self._consecutive_losses = 0
            if self._drawdown_protection:
                logger.info("Drawdown protection LIFTED — winning trade restores full size")
                self._drawdown_protection = False
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3 and not self._drawdown_protection:
                logger.warning(
                    "Drawdown protection ACTIVE — %d consecutive losses, all position sizes halved",
                    self._consecutive_losses,
                )
                self._drawdown_protection = True

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        account_balance: float,
        entry: float,
        stop: float,
        confidence: float = 1.0,
        time_scalar: float = 1.0,
    ) -> float:
        """Return volume scaled by confidence, drawdown protection, and time filter."""
        confidence_scalar = 0.5 + 0.5 * min(max(confidence, 0.0), 1.0)
        drawdown_scalar = 0.5 if self._drawdown_protection else 1.0
        risk_amount = (
            account_balance
            * self.cfg.max_risk_per_trade
            * confidence_scalar
            * drawdown_scalar
            * time_scalar
        )
        per_unit_risk = abs(entry - stop)
        if per_unit_risk < 1e-9:
            return 0.0
        volume = risk_amount / per_unit_risk
        return round(volume, 8)

    # ── Trade creation ────────────────────────────────────────────────────────

    def open_trade(
        self,
        pair: str,
        side: str,
        entry: float,
        stop: float,
        target: float,
        volume: float,
    ) -> Trade:
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:04d}"
        trade = Trade(
            id=trade_id,
            pair=pair,
            side=side,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            volume=volume,
            trailing_stop=stop,
            r_value=abs(entry - stop),
        )
        self._open_trades[trade_id] = trade
        logger.info("Opened trade %s | %s %s @ %.4f | SL %.4f TP %.4f | R=%.4f",
                    trade_id, side.upper(), pair, entry, stop, target, trade.r_value)
        return trade

    def close_trade(self, trade_id: str, exit_price: float, reason: str = "") -> Optional[Trade]:
        trade = self._open_trades.pop(trade_id, None)
        if not trade:
            return None
        trade.exit_price = exit_price
        sign = 1 if trade.side == "buy" else -1
        trade.pnl = sign * (exit_price - trade.entry_price) * trade.volume
        trade.status = "closed"
        trade.notes = reason
        self._daily_pnl += trade.pnl
        won = trade.pnl > 0
        self.record_trade_result(won)
        logger.info("Closed trade %s | exit %.4f | PnL %.4f | %s",
                    trade_id, exit_price, trade.pnl, reason)
        return trade

    def partial_close_trade(self, trade_id: str, exit_price: float) -> Optional[float]:
        """Close 50% of position at 2R. Returns the closed volume or None."""
        trade = self._open_trades.get(trade_id)
        if not trade or trade.partial_exit_done:
            return None
        half_vol = round(trade.volume / 2, 8)
        sign = 1 if trade.side == "buy" else -1
        partial_pnl = sign * (exit_price - trade.entry_price) * half_vol
        trade.volume = half_vol
        trade.partial_exit_done = True
        # Move SL to breakeven
        trade.stop_loss = trade.entry_price
        trade.trailing_stop = trade.entry_price
        trade.breakeven_set = True
        self._daily_pnl += partial_pnl
        logger.info(
            "Partial exit 50%% | %s | closed %.8f units @ %.4f | partial PnL %.4f | SL → breakeven",
            trade.pair, half_vol, exit_price, partial_pnl,
        )
        return half_vol

    # ── R-milestone management ────────────────────────────────────────────────

    def update_r_milestones(self, prices: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Returns list of (trade_id, price) for trades that just crossed 2R
        and need a partial exit. Also activates 3R trailing in-place.
        """
        needs_partial = []
        for tid, trade in list(self._open_trades.items()):
            if trade.r_value <= 0:
                continue
            price = prices.get(trade.pair)
            if price is None:
                continue

            sign = 1 if trade.side == "buy" else -1
            profit_r = sign * (price - trade.entry_price) / trade.r_value

            # 2R hit: queue partial exit (executed by caller)
            if profit_r >= 2.0 and not trade.partial_exit_done:
                logger.info(
                    "Trade %s hit 2R (%.2fR) | %s @ %.4f — queuing 50%% exit + breakeven SL",
                    tid, profit_r, trade.pair, price,
                )
                needs_partial.append((tid, price))

            # 3R hit: trail stop at 1R below current price
            if profit_r >= 3.0 and trade.partial_exit_done:
                r = trade.r_value
                if trade.side == "buy":
                    new_trail = round(price - r, 8)
                    if new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail
                        logger.info("Trade %s 3R trail | SL → %.4f (1R below %.4f)",
                                    tid, new_trail, price)
                else:
                    new_trail = round(price + r, 8)
                    if new_trail < trade.trailing_stop:
                        trade.trailing_stop = new_trail
                        logger.info("Trade %s 3R trail | SL → %.4f (1R above %.4f)",
                                    tid, new_trail, price)

        return needs_partial

    # ── ATR trailing stop ─────────────────────────────────────────────────────

    def update_trailing_stops(self, prices: Dict[str, float], atr_values: Dict[str, float]):
        for tid, trade in list(self._open_trades.items()):
            price = prices.get(trade.pair)
            atr = atr_values.get(trade.pair)
            if price is None or atr is None:
                continue
            if not self.cfg.trailing_stop:
                continue
            # Skip ATR trail once 3R logic has taken over
            if trade.partial_exit_done:
                continue
            offset = atr * self.cfg.trailing_stop_atr_mult
            if trade.side == "buy":
                new_stop = price - offset
                if new_stop > trade.trailing_stop:
                    trade.trailing_stop = round(new_stop, 8)
                    logger.debug("ATR trail updated %s SL → %.4f", tid, trade.trailing_stop)
            else:
                new_stop = price + offset
                if new_stop < trade.trailing_stop:
                    trade.trailing_stop = round(new_stop, 8)
                    logger.debug("ATR trail updated %s SL → %.4f", tid, trade.trailing_stop)

    # ── Stop/target checks ────────────────────────────────────────────────────

    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """Returns list of (trade_id, exit_price, reason) for full closes."""
        exits = []
        for tid, trade in list(self._open_trades.items()):
            price = prices.get(trade.pair)
            if price is None:
                continue
            active_stop = trade.trailing_stop if self.cfg.trailing_stop else trade.stop_loss
            if trade.side == "buy":
                if price <= active_stop:
                    exits.append((tid, price, "stop_loss"))
                elif price >= trade.take_profit and not trade.partial_exit_done:
                    exits.append((tid, price, "take_profit"))
            else:
                if price >= active_stop:
                    exits.append((tid, price, "stop_loss"))
                elif price <= trade.take_profit and not trade.partial_exit_done:
                    exits.append((tid, price, "take_profit"))
        return exits

    # ── ATR helper ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return float(sum(trs[-period:]) / period)

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> Dict[str, Trade]:
        return dict(self._open_trades)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl
