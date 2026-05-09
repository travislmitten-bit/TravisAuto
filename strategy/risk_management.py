"""
Position sizing and trade lifecycle management.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Deque, Dict, List, Optional, Tuple

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
    status: str = "open"
    exit_price: float = 0.0
    pnl: float = 0.0
    trailing_stop: float = 0.0
    notes: str = ""
    r_value: float = 0.0           # |entry - stop|
    partial_exit_done: bool = False # 50% closed at 2R
    breakeven_set: bool = False
    hold_30_active: bool = False    # 30% held after 70% take-profit exit
    wide_stop: float = 0.0          # weekly 20MA stop for hold-30 leg


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._open_trades: Dict[str, Trade] = {}
        self._daily_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._trade_counter: int = 0

        # Drawdown protection
        self._consecutive_losses: int = 0
        self._drawdown_protection: bool = False

        # Kelly criterion — stores (pnl, trade_value) for last 200 trades
        self._trade_history: Deque[Tuple[float, float]] = deque(maxlen=200)

        # Sharpe — daily PnL % for last 60 days
        self._daily_pnl_pcts: List[float] = []
        self._current_day: date = date.today()
        self._day_start_balance: float = 0.0

    # ── Daily rollover ────────────────────────────────────────────────────────

    def tick_daily(self, account_balance: float):
        """Call once per day to record daily PnL% for Sharpe calculation."""
        today = date.today()
        if today != self._current_day:
            if self._day_start_balance > 0:
                pct = self._daily_pnl / self._day_start_balance
                self._daily_pnl_pcts.append(pct)
                if len(self._daily_pnl_pcts) > 60:
                    self._daily_pnl_pcts = self._daily_pnl_pcts[-60:]
            self._daily_pnl = 0.0
            self._daily_date = today
            self._current_day = today
            self._day_start_balance = account_balance

    # ── Checks ────────────────────────────────────────────────────────────────

    def can_trade(self, account_balance: float) -> bool:
        if len(self._open_trades) >= self.cfg.max_open_trades:
            logger.warning("Max open trades reached (%d)", self.cfg.max_open_trades)
            return False
        today = date.today()
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_date = today
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
                logger.info("Drawdown protection LIFTED — winning trade restores full sizing")
                self._drawdown_protection = False
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3 and not self._drawdown_protection:
                logger.warning(
                    "Drawdown protection ACTIVE — %d consecutive losses, all sizes halved",
                    self._consecutive_losses,
                )
                self._drawdown_protection = True

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        account_balance: float,
        entry: float,
        stop: float,
        effective_risk_pct: float,
    ) -> float:
        """
        Pure risk-amount sizing. All scalars (confidence, regime, ADX, Kelly,
        Sharpe, rotation, time, drawdown) are pre-applied in effective_risk_pct.
        """
        per_unit_risk = abs(entry - stop)
        if per_unit_risk < 1e-9:
            return 0.0
        risk_amount = account_balance * effective_risk_pct
        return round(risk_amount / per_unit_risk, 8)

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
        logger.info(
            "Opened trade %s | %s %s @ %.4f | SL %.4f TP %.4f | R=%.4f",
            trade_id, side.upper(), pair, entry, stop, target, trade.r_value,
        )
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
        trade_value = trade.entry_price * trade.volume
        self._trade_history.append((trade.pnl, trade_value))
        won = trade.pnl > 0
        self.record_trade_result(won)
        logger.info(
            "Closed trade %s | exit %.4f | PnL %.4f | %s",
            trade_id, exit_price, trade.pnl, reason,
        )
        return trade

    # ── Partial exits ─────────────────────────────────────────────────────────

    def partial_close_trade(self, trade_id: str, exit_price: float) -> Optional[float]:
        """2R milestone: close 50%, move SL to breakeven. Returns closed volume."""
        trade = self._open_trades.get(trade_id)
        if not trade or trade.partial_exit_done:
            return None
        half = round(trade.volume / 2, 8)
        sign = 1 if trade.side == "buy" else -1
        self._daily_pnl += sign * (exit_price - trade.entry_price) * half
        trade.volume          = half
        trade.partial_exit_done = True
        trade.stop_loss       = trade.entry_price
        trade.trailing_stop   = trade.entry_price
        trade.breakeven_set   = True
        logger.info(
            "Partial 2R exit | %s | closed %.8f @ %.4f | SL → breakeven %.4f",
            trade.pair, half, exit_price, trade.entry_price,
        )
        return half

    def exit_70_hold_30(
        self, trade_id: str, exit_price: float, wide_stop: float
    ) -> Optional[float]:
        """
        Take-profit exit: close 70%, hold remaining 30% with wide_stop.
        Returns volume closed (70%).
        """
        trade = self._open_trades.get(trade_id)
        if not trade or trade.hold_30_active:
            return None
        close_vol = round(trade.volume * 0.70, 8)
        hold_vol  = round(trade.volume * 0.30, 8)
        sign = 1 if trade.side == "buy" else -1
        self._daily_pnl += sign * (exit_price - trade.entry_price) * close_vol
        trade.volume        = hold_vol
        trade.hold_30_active = True
        trade.wide_stop     = wide_stop
        trade.trailing_stop = wide_stop
        trade.stop_loss     = wide_stop
        logger.info(
            "Take-profit 70%% exit | %s | closed %.8f @ %.4f | holding %.8f with wide SL %.4f (weekly 20MA)",
            trade.pair, close_vol, exit_price, hold_vol, wide_stop,
        )
        return close_vol

    # ── R-milestone updates ───────────────────────────────────────────────────

    def update_r_milestones(self, prices: Dict[str, float]) -> List[Tuple[str, float]]:
        """Returns (trade_id, price) pairs that just crossed 2R → need partial exit."""
        needs_partial: List[Tuple[str, float]] = []
        for tid, trade in list(self._open_trades.items()):
            if trade.r_value <= 0 or trade.hold_30_active:
                continue
            price = prices.get(trade.pair)
            if price is None:
                continue
            sign     = 1 if trade.side == "buy" else -1
            profit_r = sign * (price - trade.entry_price) / trade.r_value

            if profit_r >= 2.0 and not trade.partial_exit_done:
                logger.info(
                    "Trade %s hit 2R (%.2fR) | %s @ %.4f — queuing 50%% exit + breakeven",
                    tid, profit_r, trade.pair, price,
                )
                needs_partial.append((tid, price))

            if profit_r >= 3.0 and trade.partial_exit_done:
                r = trade.r_value
                if trade.side == "buy":
                    new_trail = round(price - r, 8)
                    if new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail
                        logger.info("Trade %s 3R trail | SL → %.4f", tid, new_trail)
                else:
                    new_trail = round(price + r, 8)
                    if new_trail < trade.trailing_stop:
                        trade.trailing_stop = new_trail
                        logger.info("Trade %s 3R trail | SL → %.4f", tid, new_trail)

        return needs_partial

    def check_hold_exits(self, weekly_20ma: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """Check if weekly close crossed below (or above for shorts) the 20-week MA."""
        exits = []
        for tid, trade in list(self._open_trades.items()):
            if not trade.hold_30_active:
                continue
            ma = weekly_20ma.get(trade.pair)
            if ma is None:
                continue
            price = trade.wide_stop  # current MA level stored as wide_stop
            if trade.side == "buy" and ma < trade.wide_stop:
                exits.append((tid, ma, "weekly_20ma_break"))
            elif trade.side == "sell" and ma > trade.wide_stop:
                exits.append((tid, ma, "weekly_20ma_break"))
        return exits

    # ── ATR trailing stop ─────────────────────────────────────────────────────

    def update_trailing_stops(
        self,
        prices: Dict[str, float],
        atr_values: Dict[str, float],
        trail_mult: float = 1.0,
    ):
        for tid, trade in list(self._open_trades.items()):
            if trade.partial_exit_done or trade.hold_30_active:
                continue
            price = prices.get(trade.pair)
            atr   = atr_values.get(trade.pair)
            if price is None or atr is None or not self.cfg.trailing_stop:
                continue
            offset = atr * self.cfg.trailing_stop_atr_mult * trail_mult
            if trade.side == "buy":
                new_stop = price - offset
                if new_stop > trade.trailing_stop:
                    trade.trailing_stop = round(new_stop, 8)
            else:
                new_stop = price + offset
                if new_stop < trade.trailing_stop:
                    trade.trailing_stop = round(new_stop, 8)

    # ── Stop / target checks ──────────────────────────────────────────────────

    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, float, str]]:
        exits = []
        for tid, trade in list(self._open_trades.items()):
            price = prices.get(trade.pair)
            if price is None:
                continue
            if trade.hold_30_active:
                continue  # handled by check_hold_exits
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

    @property
    def trade_history(self):
        return self._trade_history

    @property
    def daily_pnl_pcts(self) -> List[float]:
        return self._daily_pnl_pcts
