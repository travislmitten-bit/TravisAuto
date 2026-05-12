"""
Position sizing and trade lifecycle management.
"""

from __future__ import annotations
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

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
    exit_mode: str = "4h"           # "4h" | "daily" | "weekly"
    exit_mode_reason: str = ""      # human-readable reason for last switch
    pyramid_count: int = 0          # pyramids added (max 2)
    original_volume: float = 0.0    # captured on first pyramid for sizing reference
    confidence: float = 0.0         # adjusted confidence at entry
    regime: str = ""                # asset regime label at entry
    entry_txid: str = ""            # Kraken txid of entry market order
    broker_stop_txid: str = ""      # Kraken txid of currently-active stop-loss order

    @property
    def exit_mode_elevated(self) -> bool:
        return self.exit_mode in ("daily", "weekly")


class RiskManager:
    def __init__(self, cfg: RiskConfig, history_path: Optional[Path] = None):
        self.cfg = cfg
        self._open_trades: Dict[str, Trade] = {}
        self._daily_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._trade_counter: int = 0
        self._history_path: Optional[Path] = Path(history_path) if history_path else None

        # Drawdown protection
        self._consecutive_losses: int = 0
        self._drawdown_protection: bool = False

        # Equity-curve drawdown (peak-to-trough)
        self._peak_balance: float = 0.0
        self._kill_switch_armed: bool = False
        self._dd_pause_logged: bool = False
        self._dd_kill_logged:  bool = False

        # Kelly criterion — stores (pnl, trade_value) for last 200 trades
        self._trade_history: Deque[Tuple[float, float]] = deque(maxlen=200)
        self._load_history()

        # Sharpe — daily PnL % for last 60 days
        self._daily_pnl_pcts: List[float] = []
        self._current_day: date = date.today()
        self._day_start_balance: float = 0.0

    # ── Persistent history ───────────────────────────────────────────────────

    def _load_history(self):
        """Hydrate the in-memory deque from the on-disk trade-history JSON."""
        if not self._history_path or not self._history_path.exists():
            return
        try:
            with self._history_path.open() as f:
                records = json.load(f)
            if not isinstance(records, list):
                logger.warning("Trade history file is not a list — ignoring")
                return
            valid_for_kelly = 0
            wins = losses = 0
            for rec in records[-200:]:
                try:
                    pnl = float(rec.get("pnl", 0.0))
                    tv  = float(rec.get("trade_value", 0.0))
                except (TypeError, ValueError):
                    continue
                if tv > 0:
                    self._trade_history.append((pnl, tv))
                    valid_for_kelly += 1
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            win_rate = (wins / valid_for_kelly) if valid_for_kelly else 0.0
            logger.warning(
                "TRADE HISTORY | loaded %d records (%d valid for Kelly) | "
                "wins=%d losses=%d win_rate=%.1f%% | source=%s",
                len(records), valid_for_kelly, wins, losses, win_rate * 100,
                self._history_path,
            )
        except Exception as e:
            logger.warning(
                "TRADE HISTORY | load failed: %s — starting fresh", e,
            )

    def _append_history(self, trade: "Trade", exit_price: float, reason: str):
        """Atomically append a completed-trade record to the history JSON."""
        if not self._history_path:
            return
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            records: list = []
            if self._history_path.exists():
                try:
                    with self._history_path.open() as f:
                        loaded = json.load(f)
                    if isinstance(loaded, list):
                        records = loaded
                except Exception as e:
                    logger.warning("History reload before append failed: %s", e)
            records.append({
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
                "trade_id":    trade.id,
                "pair":        trade.pair,
                "side":        trade.side,
                "entry_price": round(trade.entry_price, 8),
                "exit_price":  round(exit_price, 8),
                "volume":      round(trade.volume, 8),
                "pnl":         round(trade.pnl, 8),
                "trade_value": round(trade.entry_price * trade.volume, 8),
                "confidence":  round(trade.confidence, 4),
                "regime":      trade.regime,
                "reason":      reason,
            })
            tmp = self._history_path.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(records, f, indent=2)
            tmp.replace(self._history_path)
        except Exception as e:
            logger.warning("Trade history append failed: %s", e)

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
        if self._kill_switch_armed:
            return False
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

        # Equity-curve drawdown gates
        dd = self.drawdown_pct(account_balance)
        if dd >= 0.20:
            self._kill_switch_armed = True
            if not self._dd_kill_logged:
                logger.critical(
                    "KILL SWITCH ARMED | drawdown %.1f%% ≥ 20%% | peak=$%.2f current=$%.2f | "
                    "halting all new entries — manual restart required to clear",
                    dd * 100, self._peak_balance, account_balance,
                )
                self._dd_kill_logged = True
            return False
        if dd >= 0.15:
            if not self._dd_pause_logged:
                logger.warning(
                    "DRAWDOWN PAUSE | drawdown %.1f%% ≥ 15%% | peak=$%.2f current=$%.2f | "
                    "halting new entries until recovery",
                    dd * 100, self._peak_balance, account_balance,
                )
                self._dd_pause_logged = True
            return False
        return True

    # ── Equity-curve drawdown ────────────────────────────────────────────────

    def update_peak(self, account_balance: float):
        if account_balance > self._peak_balance:
            self._peak_balance = account_balance
            self._dd_pause_logged = False
            self._dd_kill_logged  = False

    def drawdown_pct(self, account_balance: float) -> float:
        if self._peak_balance <= 1e-9:
            return 0.0
        return max(0.0, (self._peak_balance - account_balance) / self._peak_balance)

    @property
    def peak_balance(self) -> float:
        return self._peak_balance

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_armed

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
        confidence: float = 0.0,
        regime: str = "",
        entry_txid: str = "",
        broker_stop_txid: str = "",
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
            confidence=confidence,
            regime=regime,
            entry_txid=entry_txid,
            broker_stop_txid=broker_stop_txid,
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
        self._append_history(trade, exit_price, reason)
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
        self,
        trade_id:  str,
        exit_price: float,
        wide_stop:  float,
        hold_pct:   float = 0.30,
    ) -> Optional[float]:
        """
        Partial take-profit exit.  hold_pct fraction is kept open with wide_stop.
        Regime-driven split: BEAR 80/20, CHOPPY 75/25, BULL 60/40, PARABOLIC 50/50.
        Returns volume closed (exit_pct portion), or None if already active.
        """
        trade = self._open_trades.get(trade_id)
        if not trade or trade.hold_30_active:
            return None
        exit_pct  = round(1.0 - hold_pct, 8)
        close_vol = round(trade.volume * exit_pct, 8)
        hold_vol  = round(trade.volume * hold_pct, 8)
        sign = 1 if trade.side == "buy" else -1
        self._daily_pnl += sign * (exit_price - trade.entry_price) * close_vol
        trade.volume         = hold_vol
        trade.hold_30_active = True
        trade.wide_stop      = wide_stop
        trade.trailing_stop  = wide_stop
        trade.stop_loss      = wide_stop
        logger.info(
            "Partial TP %.0f%%/%.0f%% | %s | closed %.8f @ %.4f | "
            "holding %.8f with wide SL %.4f (weekly 20MA)",
            exit_pct * 100, hold_pct * 100,
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

    # ── Pyramiding ────────────────────────────────────────────────────────────

    def add_pyramid(
        self,
        trade_id: str,
        price:    float,
        volume:   float,
        new_stop: float,
    ) -> Optional[Trade]:
        """Add to a winning position. Max 2 pyramids. Stop only moves toward profit."""
        trade = self._open_trades.get(trade_id)
        if not trade or trade.pyramid_count >= 2:
            return None
        if trade.original_volume == 0.0:
            trade.original_volume = trade.volume
        trade.pyramid_count += 1
        trade.volume += volume
        if trade.side == "buy":
            trade.stop_loss    = max(trade.stop_loss,    new_stop)
            trade.trailing_stop = max(trade.trailing_stop, new_stop)
        else:
            trade.stop_loss    = min(trade.stop_loss,    new_stop)
            trade.trailing_stop = min(trade.trailing_stop, new_stop)
        logger.info(
            "PYRAMID #%d | %s | +%.8f @ %.4f | SL → %.4f | total_vol=%.8f",
            trade.pyramid_count, trade.pair, volume, price, new_stop, trade.volume,
        )
        return trade

    # ── ATR trailing stop ─────────────────────────────────────────────────────

    def tighten_trailing_stop(
        self,
        trade_id: str,
        price: float,
        atr: float,
        tight_mult: float = 0.5,
    ):
        """Tighten stop on a specific trade to tight_mult × ATR_mult × ATR."""
        trade = self._open_trades.get(trade_id)
        if not trade or not self.cfg.trailing_stop:
            return
        offset = atr * self.cfg.trailing_stop_atr_mult * tight_mult
        if trade.side == "buy":
            new_stop = round(price - offset, 8)
            if new_stop > trade.trailing_stop:
                trade.trailing_stop = new_stop
        else:
            new_stop = round(price + offset, 8)
            if new_stop < trade.trailing_stop:
                trade.trailing_stop = new_stop

    def update_trailing_stops(
        self,
        prices: Dict[str, float],
        atr_values: Dict[str, float],
        trail_mult: Union[float, Dict[str, float]] = 1.0,
    ):
        for tid, trade in list(self._open_trades.items()):
            if trade.partial_exit_done or trade.hold_30_active:
                continue
            price = prices.get(trade.pair)
            atr   = atr_values.get(trade.pair)
            if price is None or atr is None or not self.cfg.trailing_stop:
                continue
            mult = (
                trail_mult.get(trade.pair, 1.0)
                if isinstance(trail_mult, dict)
                else trail_mult
            )
            offset = atr * self.cfg.trailing_stop_atr_mult * mult
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
                elif price >= trade.take_profit and not trade.partial_exit_done \
                        and not trade.exit_mode_elevated:
                    exits.append((tid, price, "take_profit"))
            else:
                if price >= active_stop:
                    exits.append((tid, price, "stop_loss"))
                elif price <= trade.take_profit and not trade.partial_exit_done \
                        and not trade.exit_mode_elevated:
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
