"""
TravisAuto Dashboard
====================
Terminal dashboard using Rich. Run alongside the bot:
    python dashboard/dashboard.py

Reads live data from Kraken + introspects the bot's state via a shared
state file (dashboard_state.json) written by main.py, or runs standalone
to display market data and trendline signals.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)

from config.config import CONFIG
from kraken.api import KrakenAPI
from strategy.trendline import generate_signals, Signal, detect_trendlines
from strategy.risk_management import RiskManager

console = Console()

STATE_FILE = Path("logs/dashboard_state.json")

SIGNAL_COLORS = {
    Signal.BUY_BOUNCE:  "green",
    Signal.BUY_BREAK:   "bright_green",
    Signal.SELL_BOUNCE: "red",
    Signal.SELL_BREAK:  "bright_red",
    Signal.NONE:        "white",
}

SIGNAL_ICONS = {
    Signal.BUY_BOUNCE:  "^ BOUNCE",
    Signal.BUY_BREAK:   ">> BREAK",
    Signal.SELL_BOUNCE: "v BOUNCE",
    Signal.SELL_BREAK:  "<< BREAK",
}


class Dashboard:
    def __init__(self):
        self.kraken = KrakenAPI(
            api_key=CONFIG.kraken.api_key,
            api_secret=CONFIG.kraken.api_secret,
        )
        self._prices: Dict[str, float] = {}
        self._signals: list = []
        self._open_trades: list = []
        self._daily_pnl: float = 0.0
        self._balance: float = 0.0
        self._last_update: str = "—"
        self._errors: List[str] = []

    # ── Data refresh ──────────────────────────────────────────────────────────

    def _refresh(self):
        self._signals = []
        for pair in CONFIG.pairs:
            try:
                price = self.kraken.get_mid_price(pair)
                self._prices[pair] = price
            except Exception as e:
                self._errors.append(f"{pair}: {e}")

            try:
                raw = self.kraken.get_ohlcv(pair, interval=CONFIG.interval)
                raw = raw[-CONFIG.trendline.lookback_candles:]
                opens   = np.array([float(c[1]) for c in raw])
                highs   = np.array([float(c[2]) for c in raw])
                lows    = np.array([float(c[3]) for c in raw])
                closes  = np.array([float(c[4]) for c in raw])
                volumes = np.array([float(c[6]) for c in raw])

                sigs = generate_signals(
                    pair=pair,
                    opens=opens, highs=highs, lows=lows,
                    closes=closes, volumes=volumes,
                    cfg=CONFIG.trendline,
                    rr_ratio=CONFIG.risk.default_rr_ratio,
                )
                self._signals.extend(sigs)
            except Exception as e:
                self._errors.append(f"Signal {pair}: {e}")

        # Load bot state if available
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                self._open_trades = state.get("open_trades", [])
                self._daily_pnl   = state.get("daily_pnl", 0.0)
                self._balance     = state.get("balance", 0.0)
            except Exception:
                pass

        self._last_update = datetime.now().strftime("%H:%M:%S")
        self._errors = self._errors[-5:]  # keep last 5

    # ── Render helpers ────────────────────────────────────────────────────────

    def _make_header(self) -> Panel:
        mode = "[red bold]LIVE[/]" if not CONFIG.dry_run else "[yellow]DRY RUN[/]"
        title = Text.assemble(
            ("TravisAuto ", "bold cyan"),
            ("| Tori Trendline Strategy  ", "white"),
            (mode, ""),
            ("  updated: ", "dim"),
            (self._last_update, "dim"),
        )
        return Panel(title, box=box.HORIZONTALS)

    def _make_prices_table(self) -> Table:
        t = Table(title="Market Prices", box=box.SIMPLE, expand=True)
        t.add_column("Pair", style="cyan")
        t.add_column("Price", justify="right")
        t.add_column("Signal", justify="center")
        t.add_column("Conf", justify="right")
        t.add_column("SL", justify="right")
        t.add_column("TP", justify="right")

        sig_map = {s.pair: s for s in self._signals}
        for pair in CONFIG.pairs:
            price = self._prices.get(pair, 0.0)
            sig = sig_map.get(pair)
            if sig:
                color = SIGNAL_COLORS.get(sig.signal, "white")
                sig_label = f"[{color}]{SIGNAL_ICONS.get(sig.signal, sig.signal.value)}[/]"
                conf_str  = f"[{color}]{sig.confidence:.0%}[/]"
                sl_str    = f"{sig.suggested_stop:.4f}"
                tp_str    = f"{sig.suggested_target:.4f}"
            else:
                sig_label = "[dim]—[/]"
                conf_str  = "[dim]—[/]"
                sl_str    = "—"
                tp_str    = "—"
            t.add_row(pair, f"{price:,.4f}", sig_label, conf_str, sl_str, tp_str)
        return t

    def _make_trades_table(self) -> Table:
        t = Table(title="Open Trades", box=box.SIMPLE, expand=True)
        t.add_column("ID", style="dim")
        t.add_column("Pair", style="cyan")
        t.add_column("Side")
        t.add_column("Entry", justify="right")
        t.add_column("SL", justify="right")
        t.add_column("TP", justify="right")
        t.add_column("Vol", justify="right")

        if not self._open_trades:
            t.add_row("—", "No open trades", "", "", "", "", "")
        else:
            for tr in self._open_trades:
                side_color = "green" if tr.get("side") == "buy" else "red"
                t.add_row(
                    tr.get("id", "?"),
                    tr.get("pair", "?"),
                    f"[{side_color}]{tr.get('side','').upper()}[/]",
                    f"{tr.get('entry_price', 0):.4f}",
                    f"{tr.get('stop_loss', 0):.4f}",
                    f"{tr.get('take_profit', 0):.4f}",
                    f"{tr.get('volume', 0):.6f}",
                )
        return t

    def _make_stats_panel(self) -> Panel:
        pnl_color = "green" if self._daily_pnl >= 0 else "red"
        lines = [
            f"Balance:    [cyan]{self._balance:,.2f} USD[/]",
            f"Daily PnL:  [{pnl_color}]{self._daily_pnl:+.4f}[/]",
            f"Open Trades: {len(self._open_trades)} / {CONFIG.risk.max_open_trades}",
            f"Max Risk/Trade: {CONFIG.risk.max_risk_per_trade:.0%}",
            f"Interval: {CONFIG.interval}m | Pairs: {len(CONFIG.pairs)}",
        ]
        return Panel("\n".join(lines), title="Stats", box=box.SIMPLE)

    def _make_errors_panel(self) -> Panel:
        if not self._errors:
            content = "[dim]No errors[/]"
        else:
            content = "\n".join(f"[red]{e}[/]" for e in self._errors)
        return Panel(content, title="Errors", box=box.SIMPLE)

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._make_header(), size=3, name="header"),
            Layout(name="body"),
            Layout(self._make_errors_panel(), size=6, name="errors"),
        )
        layout["body"].split_row(
            Layout(self._make_prices_table(), name="prices"),
            Layout(name="right"),
        )
        layout["right"].split_column(
            Layout(self._make_stats_panel(), name="stats"),
            Layout(self._make_trades_table(), name="trades"),
        )
        return layout

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, refresh_seconds: int = 30):
        console.print("[bold cyan]TravisAuto Dashboard[/] starting...\n")
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                self._refresh()
                live.update(self._build_layout())
                time.sleep(refresh_seconds)


def main():
    refresh = int(os.getenv("DASH_REFRESH", "30"))
    Dashboard().run(refresh_seconds=refresh)


if __name__ == "__main__":
    main()
