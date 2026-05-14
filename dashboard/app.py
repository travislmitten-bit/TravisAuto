"""
dashboard/app.py — Read-only Flask dashboard for TravisAuto.

Runs as a SEPARATE systemd service from the trading bot. All data comes from:
  - `journalctl -u travisauto` (parsed by regex)
  - `~/TravisAuto/data/trade_history.json` (closed trades)
  - `systemctl show travisauto` (uptime)

Never touches Kraken, never imports bot modules, can't break trading.
Auto-refreshes every 30s. No auth (per spec). Listens on 0.0.0.0:8080.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template_string

REPO = Path("/root/TravisAuto")
HISTORY_FILE = REPO / "data" / "trade_history.json"
CACHE_TTL = 5.0      # seconds — coalesce rapid refreshes


app = Flask(__name__)
_cache: dict = {"text": "", "ts": 0.0}


def _journal(since: str = "12 hours ago", lines: int = 8000) -> str:
    """journalctl pull with simple TTL cache."""
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL and _cache["text"]:
        return _cache["text"]
    try:
        r = subprocess.run(
            ["journalctl", "-u", "travisauto", "--since", since,
             "--no-pager", "-n", str(lines)],
            capture_output=True, text=True, timeout=10,
        )
        _cache["text"] = r.stdout
        _cache["ts"]   = now
        return r.stdout
    except Exception as e:
        return f"# journal error: {e}\n"


def _bot_uptime() -> str:
    try:
        r = subprocess.run(
            ["systemctl", "show", "travisauto",
             "--property=ActiveEnterTimestamp,ActiveState,MainPID"],
            capture_output=True, text=True, timeout=5,
        )
        info = dict(line.split("=", 1) for line in r.stdout.strip().split("\n") if "=" in line)
        ts_raw = info.get("ActiveEnterTimestamp", "")
        state  = info.get("ActiveState", "?")
        pid    = info.get("MainPID", "?")
        if not ts_raw or len(ts_raw.split()) < 4:
            return f"state={state} pid={pid}"
        parts = ts_raw.split()
        try:
            ts = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            up = datetime.now(timezone.utc) - ts
            h = int(up.total_seconds() // 3600)
            m = int((up.total_seconds() % 3600) // 60)
            return f"{h}h {m}m · state={state} · pid={pid}"
        except Exception:
            return f"{ts_raw} state={state} pid={pid}"
    except Exception as e:
        return f"err: {e}"


# ── Log parsers ──────────────────────────────────────────────────────────────

def _latest(pattern: re.Pattern, text: str):
    """Return the last regex match in text or None."""
    last = None
    for line in text.split("\n"):
        m = pattern.search(line)
        if m:
            last = m
    return last


_BAL_RE      = re.compile(r"(?:STARTUP BALANCE|BALANCE CONFIRMED) \| free ZUSD = \$([\d.]+)")
_REGIME_RE   = re.compile(r"ASSET_REGIME \| (.+)$")
_SCAN_RE     = re.compile(
    r"DataScanner: composite=([\d.]+) \[(\w+)\] \| sources=(\d+/\d+) \| (.+)$"
)
_STAKING_RE  = re.compile(r"total accumulated: \$([\d.]+)")
_REJ_RE      = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.\d]*\+?\d*|\w{3} \d{2} \d{2}:\d{2}:\d{2}).*?"
    r"TravisAuto: (SIGNAL_REJECTED \| .+)$"
)
_OPEN_RE     = re.compile(
    r"Opened trade (T\d+) \| (BUY|SELL) (\S+) @ ([\d.]+) \| SL ([\d.]+) TP ([\d.]+) \| R=([\d.]+)"
)
_CLOSE_RE    = re.compile(r"Closed trade (T\d+) \|")
_KILL_RE     = re.compile(r"KILL SWITCH ARMED \| drawdown ([\d.]+)%")


def _latest_balance(text: str) -> str:
    m = _latest(_BAL_RE, text)
    return m.group(1) if m else "—"


def _latest_regimes(text: str) -> dict:
    m = _latest(_REGIME_RE, text)
    if not m:
        return {}
    out = {}
    for chunk in m.group(1).split(" | "):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _latest_composite(text: str) -> dict:
    m = _latest(_SCAN_RE, text)
    if not m:
        return {"composite": "—", "label": "—", "sources": "—", "raw": "(no scanner cycle yet)"}
    return {
        "composite": m.group(1),
        "label":     m.group(2),
        "sources":   m.group(3),
        "raw":       m.group(4),
    }


def _latest_staking(text: str) -> str:
    m = _latest(_STAKING_RE, text)
    return m.group(1) if m else "0.0000"


def _kill_switch_state(text: str) -> str:
    m = _latest(_KILL_RE, text)
    return f"ARMED @ {m.group(1)}%" if m else "inactive"


def _recent_rejections(text: str, n: int = 10) -> list:
    """Return last n SIGNAL_REJECTED lines (deduplicating consecutive identical pair+reason)."""
    rows = []
    for line in text.split("\n"):
        m = _REJ_RE.search(line)
        if m:
            rows.append((m.group(1), m.group(2)))
    out = []
    last_sig = None
    for ts, body in reversed(rows):
        m2 = re.search(r"SIGNAL_REJECTED \| (\S+) \| reason=(\S+)", body)
        sig = (m2.group(1), m2.group(2)) if m2 else None
        if sig and sig == last_sig:
            continue
        last_sig = sig
        # Trim leading "TravisAuto: " etc. - body already starts with SIGNAL_REJECTED
        # Extract pair + reason + detail
        pair = m2.group(1) if m2 else "?"
        full_after_reason = re.sub(r"^SIGNAL_REJECTED \| \S+ \| ", "", body)
        out.append({"ts": ts, "pair": pair, "detail": full_after_reason})
        if len(out) >= n:
            break
    return out


def _open_positions(text: str) -> list:
    """Reconstruct currently-open positions by matching opens that have no close."""
    open_map: dict = {}
    for line in text.split("\n"):
        m = _OPEN_RE.search(line)
        if m:
            tid = m.group(1)
            open_map[tid] = {
                "id":     tid,
                "side":   m.group(2),
                "pair":   m.group(3),
                "entry":  m.group(4),
                "stop":   m.group(5),
                "target": m.group(6),
                "r":      m.group(7),
            }
            continue
        m = _CLOSE_RE.search(line)
        if m and m.group(1) in open_map:
            del open_map[m.group(1)]
    return list(open_map.values())


def _last_trades(n: int = 5) -> list:
    try:
        if not HISTORY_FILE.exists():
            return []
        with HISTORY_FILE.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data[-n:][::-1]   # newest first
    except Exception:
        return []


# ── HTML template ────────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>TravisAuto Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
         background: #0d1117; color: #c9d1d9; max-width: 1200px;
         margin: 0 auto; padding: 24px 28px; font-size: 13px; line-height: 1.5; }
  h1 { color: #58a6ff; margin: 0; font-size: 22px; font-weight: 600; }
  h2 { color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 6px;
       margin: 28px 0 10px 0; font-size: 14px; font-weight: 500; text-transform: uppercase;
       letter-spacing: 0.6px; }
  .meta { color: #8b949e; font-size: 11px; margin: 4px 0 22px 0; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .grid-4-narrow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .card { background: #161b22; border: 1px solid #30363d; padding: 10px 14px;
          border-radius: 6px; }
  .card .label { color: #8b949e; font-size: 10px; text-transform: uppercase;
                 letter-spacing: 0.7px; }
  .card .value { color: #f0f6fc; font-size: 18px; margin-top: 4px; font-weight: 600; }
  .card.bull   .value { color: #3fb950; }
  .card.bear   .value { color: #f85149; }
  .card.choppy .value { color: #d29922; }
  .card.parabolic .value { color: #a371f7; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #21262d;
           font-size: 12px; }
  th { color: #8b949e; font-weight: 500; text-transform: uppercase;
       font-size: 10px; letter-spacing: 0.6px; }
  tr:hover td { background: #161b22; }
  .pnl-pos { color: #3fb950; font-weight: 500; }
  .pnl-neg { color: #f85149; font-weight: 500; }
  pre { background: #0d1117; border: 1px solid #21262d; padding: 8px 12px;
        font-size: 11px; overflow-x: auto; border-radius: 4px;
        white-space: pre-wrap; word-break: break-all; }
  .footer { color: #6e7681; font-size: 10px; margin-top: 32px; text-align: center; }
  .empty { color: #6e7681; font-style: italic; padding: 8px 0; }
</style>
</head>
<body>
  <h1>TravisAuto</h1>
  <div class="meta">Refreshes every 30s · Generated {{ now }} UTC · Bot {{ uptime }}</div>

  <div class="grid-4">
    <div class="card">
      <div class="label">Free ZUSD</div>
      <div class="value">${{ balance }}</div>
    </div>
    <div class="card">
      <div class="label">Composite</div>
      <div class="value">{{ composite.composite }} <span class="meta">[{{ composite.label }}]</span></div>
    </div>
    <div class="card">
      <div class="label">Staking Total</div>
      <div class="value">${{ staking }}</div>
    </div>
    <div class="card">
      <div class="label">Kill Switch</div>
      <div class="value" style="color:{% if 'ARMED' in kill_switch %}#f85149{% else %}#3fb950{% endif %}">{{ kill_switch }}</div>
    </div>
  </div>

  <h2>Regime per Asset</h2>
  <div class="grid-4-narrow">
    {% for pair, regime in regimes.items() %}
      <div class="card {{ regime|lower }}">
        <div class="label">{{ pair }}</div>
        <div class="value">{{ regime }}</div>
      </div>
    {% else %}
      <div class="card"><div class="empty">(awaiting first regime cycle)</div></div>
    {% endfor %}
  </div>

  <h2>Open Positions ({{ open_positions|length }})</h2>
  {% if open_positions %}
    <table>
      <thead><tr><th>ID</th><th>Side</th><th>Pair</th><th>Entry</th><th>Stop</th><th>Target</th><th>R-value</th></tr></thead>
      <tbody>
      {% for p in open_positions %}
        <tr>
          <td>{{ p.id }}</td><td>{{ p.side }}</td><td>{{ p.pair }}</td>
          <td>${{ p.entry }}</td><td>${{ p.stop }}</td>
          <td>${{ p.target }}</td><td>{{ p.r }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="empty">no open positions</div>
  {% endif %}

  <h2>Last 10 Signal Rejections (deduped)</h2>
  <table>
    <thead><tr><th width="180">When</th><th width="80">Pair</th><th>Detail</th></tr></thead>
    <tbody>
    {% for r in rejections %}
      <tr>
        <td>{{ r.ts }}</td>
        <td>{{ r.pair }}</td>
        <td>{{ r.detail }}</td>
      </tr>
    {% else %}
      <tr><td colspan="3" class="empty">no rejections recorded</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Last 5 Closed Trades</h2>
  <table>
    <thead>
      <tr><th>Time</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th>
          <th>Vol</th><th>P&L</th><th>Reason</th></tr>
    </thead>
    <tbody>
    {% for t in trades %}
      <tr>
        <td>{{ t.timestamp }}</td>
        <td>{{ t.pair }}</td>
        <td>{{ t.side }}</td>
        <td>${{ "%.4f"|format(t.entry_price) }}</td>
        <td>${{ "%.4f"|format(t.exit_price) }}</td>
        <td>{{ "%.8f"|format(t.volume) }}</td>
        <td class="{% if t.pnl >= 0 %}pnl-pos{% else %}pnl-neg{% endif %}">${{ "%+.4f"|format(t.pnl) }}</td>
        <td>{{ t.reason }}</td>
      </tr>
    {% else %}
      <tr><td colspan="8" class="empty">no closed trades yet</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Data Scanner Detail (latest cycle)</h2>
  <pre>{{ composite.raw }}</pre>

  <div class="footer">TravisAuto Dashboard · {{ now }} UTC · journalctl-backed read-only view</div>
</body>
</html>
"""


@app.route("/")
def index():
    text = _journal()
    return render_template_string(
        HTML,
        now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        uptime=_bot_uptime(),
        balance=_latest_balance(text),
        regimes=_latest_regimes(text),
        composite=_latest_composite(text),
        staking=_latest_staking(text),
        kill_switch=_kill_switch_state(text),
        rejections=_recent_rejections(text, 10),
        open_positions=_open_positions(text),
        trades=_last_trades(5),
    )


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
