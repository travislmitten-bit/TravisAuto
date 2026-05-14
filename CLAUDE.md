# TravisAuto V1

Live Kraken trading bot — 4H trendline strategy across XBTUSD, SOLUSD, TAOUSD, LINKUSD.

## Where things live

| | |
|---|---|
| **Droplet** | `68.183.112.176` (DigitalOcean, $4 tier) |
| **Service** | `travisauto.service` (systemd, `User=root`, `WorkingDirectory=/root/TravisAuto`) |
| **Dashboard service** | `travisauto-dashboard.service` (Flask read-only view) |
| **Dashboard URL** | http://68.183.112.176:8080 (auto-refresh 30s, no auth) |
| **GitHub** | https://github.com/travislmitten-bit/TravisAuto |
| **Owner** | Travis Mitten, British Columbia, Canada — Pacific Time |

## Claude Dashboard Integration

- **Tunnel service:** `cloudflared.service` (systemd, auto-starts on reboot)
- **Current URL:** https://prepare-known-deemed-novelty.trycloudflare.com
- **Note:** URL changes on every restart. To get current URL run:
  `journalctl -u cloudflared -n 50 --no-pager | grep trycloudflare.com | tail -1`
- Claude fetches this URL on demand to analyse bot state, trades, and rejections

## Strategy

**Tori 4H trendline strategy.** Entry on:
- **Bounce** — price within 0.3% of a confirmed support/resistance line + rejection wick + volume ≥ 1.2× 20-bar avg
- **Break** — close ≥ 0.2% beyond the line with volume confirmation

Filters: ≥ 4 touchpoints required for high-conviction lines (BULL allows 2 minimum; LINKUSD has a per-pair override of 2 in `PAIR_MIN_TOUCHES_OVERRIDE` because its pivot structure is sparse), minimum 2.4% profit-to-target from entry, ADX ≥ 20.

## Regime detection

Per-asset, hourly (`REGIME_TTL=3600`). Order of overrides (first match wins):

1. **Manual env** — `{PAIR}_REGIME_OVERRIDE=BULL` skips all detection
2. **Native** — `detect_regime()`: 10+ consecutive daily closes vs 200-day SMA
3. **Responsive BULL** (`detect_responsive_bull`) — fires when ALL three hold:
   - `close > SMA50`
   - SMA50 strictly rising each of last 5 days
   - `max(closes[-7:]) > max(closes[-14:-7])`
4. **4H HH/HL structure** — 10 consecutive 4H candles with strict higher highs + higher lows
5. **MA uptrend** — `price > SMA50`, `SMA30 > SMA50`, `price[-1] > price[-7] > price[-14]`

| Regime | min_conf | max_risk | trail_mult | hold_pct (exit/hold) |
|--------|----------|----------|-----------|---------------------|
| BULL | 0.65 | 1.5% | 2.0× ATR | 0.40 (60/40) |
| BEAR | 0.85 | 0.5% | 1.0× ATR | 0.20 (80/20) |
| CHOPPY | 0.50 | 1.0% | 1.0× ATR | 0.25 (75/25) |
| PARABOLIC | 0.45 | 2.0% | 2.5× ATR | 0.50 (50/50) |

CLARITY-Act catalyst window (May 13–15 2026 EST) raises `conf_floor=0.72` and sizes ×1.25.

## Position sizing chain

```
volume = (balance × effective_risk_pct) / |entry - stop|

effective_risk_pct = base_risk
                   × asset_params.size_scalar
                   × adx_size_scalar
                   × rotation_scalar
                   × time_scalar           (0.5 if 11pm–5am EST)
                   × drawdown_scalar       (0.5 after 3 consecutive losses)
                   × sharpe_scalar         (0.5 if 30d Sharpe < 1.0)
                   × scan_scalar           (composite > 70 → 1.2, < 40 → 0.7)
                   × primed_scalar         (1.5 if BB-compression PRIMED)
                   × social_scalar         (1.25 if LunarCrush AltRank +20)
                   × corr_scalar           (0.60 if r > 0.75 with open pos)
                   × ribbon / ribbon_score / dom / vwap / catalyst / wyckoff
                   × rsi_bear_scalar / ls_scalar / sector_rot_scalar
                   × event_scalar          (CLARITY window)
```

`base_risk = max(kelly_fraction(history), asset_params.max_risk_pct)` capped at the regime's `max_risk_pct`. Kelly hydrates from `data/trade_history.json` on startup.

## Exits — multi-timeframe

- **Default** — 4H trailing stop, ATR × `trail_mult`
- **Upgrade to daily** — ≥ 5% in 24h with no adverse 4h candle
- **Upgrade to weekly** — ≥ 15% in 48h
- **Daily revert to 4H** — daily close crosses the daily support trendline
- **Weekly close** — weekly close below the weekly support trendline → market close

Partials:
- **2R** → close 50%, SL → breakeven, broker stop replaced at breakeven
- **TP hit** → regime-driven split (60/40 BULL, 80/20 BEAR, …), held leg uses weekly 20MA as wide stop
- **Hold-30 leg** → exits when weekly close crosses the 20-week MA

All exits — partial and full — place real Kraken market sells. Internal state only mutates after broker-side success.

## Data scanner — 6 sources, 4h cache

| # | Source | Endpoint | Status |
|---|--------|----------|--------|
| 1 | Fear & Greed | `api.alternative.me/fng/` | live |
| 2 | Funding rate | `futures.kraken.com/.../tickers` | live |
| 3 | Momentum (24h) | `coingecko.com/.../simple/price` | live |
| 4 | On-chain BTC | `api.blockchain.info/stats` | live |
| 5 | Mempool fees | `mempool.space/.../recommended` | live |
| 6 | Solana TPS | `api.mainnet-beta.solana.com` | live |
| — | Taostats | `taostats.io/api/v1/stats` | needs key |
| — | LunarCrush | falls back to CoinGecko community data | live |
| — | BTC dominance | CoinGecko `/global` with 3-retry backoff | live |
| — | Open interest | OKX `/rubik/.../open-interest-volume` | live |
| — | ETF flow | CoinGlass → Glassnode → N/A | currently N/A (needs CoinGlass key) |
| — | L/S ratio | CoinGlass per-pair; TAO/LINK fixed at 1.0 | currently degraded to neutral |
| — | NUPL | Glassnode `/indicators/nupl` | needs key |
| — | Smart money divergence | CoinGecko 7d chart | live |

Composite weights: `fear_greed 0.25, funding 0.20, momentum 0.20, on_chain 0.15, mempool 0.10, solana 0.10`. Plus additive: `etf_adj (±8), oi_adj (±10), ls_comp_adj (±10)`.

## V8 indicators (entry confirmation + exit tightening)

Pure-OHLCV, no external data:
- **OBV hidden divergence** (+0.08 conf on bull; tighten trail 0.7× on bear)
- **RSI hidden divergence** (Wilder 14)
- **MACD 12/26/9 histogram** (expanding +0.07 conf; contracting tightens trail to 0.7×)
- **Ichimoku** (above-cloud +0.06 conf; TK cross above cloud +0.08; below-cloud blocks longs)
- **Wyckoff spring** (0.5% breach + 2× volume + 3 prior tests + 2-bar recovery; 48h per-pair cooldown; +1.5× size override)
- **SMC Order Blocks** (last bearish candle before 10% move; +0.10 conf when price returns)
- **Fair Value Gaps** (logged as TP magnets)
- **Fibonacci** (61.8% bounce +0.05 conf; 127/161/261% extensions as TP targets)
- **EMA ribbon** (8/13/21/34/55/89/144/233 on 4H; bull-fanning +0.08; bear-fanning blocks longs; 8/21 cross tightens trail 0.5×)
- **BB compression** (4H 20-period; new 30-day low → PRIMED → +50% size on next signal; consumed on signal)

## Risk management

- **Kill switch** — 20% peak-to-trough drawdown → `_kill_switch_armed=True`, no further entries, **manual restart required**
- **Pause** — 15% drawdown → blocks new entries, lifts when balance recovers above 85% of peak
- **Drawdown protection** — 3 consecutive losses → all sizes 0.5× until a winning trade
- **Daily loss limit** — 3% per UTC day → no new entries
- **Max open trades** — 3 simultaneous
- **Low-liquidity window** — 11pm–5am EST → sizes 0.5×
- **ZUSD sufficiency** — `available_zusd = free_zusd − Σ(open_position entry × volume)` checked before every entry; rejects with `INSUFFICIENT_ZUSD` reason
- **Startup reconciliation** — on every restart fetches Kraken open orders, verifies broker stops on tracked positions, installs missing stops, logs `STARTUP_RECONCILIATION | N positions | N verified | N placed`
- **Broker stop confirmation** — after entry market fill, places `stop-loss` market-on-trigger, polls `QueryOrders` for `status ∈ {open, pending}`. If not confirmed → emergency market close of the entry. **Never holds an unprotected position.**
- **Held-leg coverage** — after 2R partial / TP partial, a fresh broker stop is installed at the new level (breakeven post-2R, weekly-20MA post-TP)

## Alerts — Telegram

Bot: **@TravisAutoALbetBot** · Chat ID: **6482161316**

Tokens stored in `.env` as `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (gitignored). Disabled cleanly if either is empty.

Events that alert:
- Bot start / stop
- Position opened (`OPENED side pair vol entry stop target`)
- Position closed (`CLOSED pair exit P&L reason`) — covers all 4 exit paths incl. hold-30 leg
- Kill switch armed (transition-detected; fires exactly once per process)
- CLARITY-Act catalyst window activated (transition-detected)
- Unprotected entry auto-closed (stop confirmation failed)
- Emergency-close failures (manual intervention required)

## Testing

```bash
python3 tests/test_all_systems.py       # 16/16 system — trendline, EMA, VWAP, MACD,
                                         # Ichimoku, OBV, RSI, Kelly, composite,
                                         # sizing, BULL params, 60/40 split, kill
                                         # switch, profit threshold, dry-run exec,
                                         # responsive bull
python3 tests/test_stress.py            # 7/7 stress — 15% pause, 20% kill, 3-loss
                                         # drawdown, all-sources-fail, rate limiter,
                                         # history hydration, order timeout
```

Both suites must be **16/16 + 7/7 green** before any merge or deploy.

## Deployment

```bash
# Local — never deploy from a dirty branch
python3 -m py_compile main.py kraken/api.py strategy/*.py
python3 tests/test_all_systems.py
python3 tests/test_stress.py

# Deploy — rsync excluding .env (droplet has its own)
rsync -av --exclude='.env' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/trade_history.json' \
  /Users/travislmitten/Desktop/TravisAuto/ \
  root@68.183.112.176:/root/TravisAuto/

# Restart bot — rate limiter enforces 60s startup grace
ssh -i ~/.ssh/id_ed25519 root@68.183.112.176 "systemctl restart travisauto"
```

Per-private-call rate budget: **3s min interval, 15 calls/minute cap**, enforced in `kraken/api.py:_PrivateRateLimiter`.

## Critical rules

1. **NEVER restart the bot more than once per hour** — repeated private-API calls during a restart trigger Kraken's account-level `EGeneral:Temporary lockout`, which can last hours and blocks all live trading. The 60s startup grace exists to mitigate this; don't override it.
2. **NEVER commit `.env` or any API key to git** — `.env` is gitignored. The token in `notifications/telegram.py` setup docs is example-format only.
3. **NEVER use destructive Kraken endpoints (`CancelAll`) from ad-hoc scripts** — the bot's `_broker_close_position` is the single source of truth for cancellation+close.
4. **NEVER bypass the `_balance_confirmed` gate.** A new TravisAutoBot process starts with `_account_balance=$0` and refuses to trade until `_update_balance` succeeds at least once. This is intentional — protects against position-sizing off a stale or hardcoded balance.
5. **NEVER deploy with failing tests.** 16/16 + 7/7 is the merge gate.
6. **Test live changes with the TEST_TRADE one-shot first** — `TEST_TRADE=true` in `.env` runs a SELL→BUY round-trip at 0.00015 BTC notional, then auto-flips itself to `false`. Use this after any change touching `KrakenAPI` or `_execute_trade`.

## Current limitations (known)

- **ETF flow** — CoinGlass returns no usable list data without an API key (`public/v2/etf/list` upstream-gated in 2026); chain falls through to N/A. Add `COINGLASS_API_KEY` to `.env` once obtained.
- **L/S ratio** — CoinGlass `long_short_ratio` returns 500 errors at the moment; TAO/LINK have no perpetual market and always sit at neutral 1.0. All 4 pairs currently neutral, no composite adjustment.
- **NUPL** — requires `GLASSNODE_API_KEY` (paid). Currently disabled.
- **Open-position persistence** — `_open_trades` lives in memory; on crash the bot does not reconstruct them from disk. Broker-side stops survive (catastrophic protection). `trade_history.json` only stores closed trades.
- **Trailing stops** — updated in-memory every 30s but NOT continuously synced to Kraken (would exceed the 15/min rate budget). Broker stop only refreshes after partial exits. The in-bot trail is the tighter exit; the broker stop is catastrophic backstop.

## Roadmap — Elite Tier additions

Pending, ordered by priority:
1. **Pyramiding winners** — already wired (`_check_pyramids`); needs live trade history to validate 2R/4R add behaviour under MACD-expanding gate
2. **Token unlock calendar** — schedule-driven volatility windows for SOL/TAO/LINK (vest cliffs, validator unlocks)
3. **Walk-forward optimization** — rolling backtest of confidence thresholds + ATR multipliers, refit monthly
4. **Reinforcement learning layer** — on top of the trendline strategy, learns regime × indicator-combo → expected R, used as an additional filter

## TravisAuto V2 (planned, separate repo)

- Simplified daily-candle trend-following on US large-cap stocks
- Alpaca brokerage API
- Daily timeframe, weekly evaluation cycle
- No leverage, no shorting
- Designed for taxable accounts in Canada

## Commercial path (longer-term)

Education + software model based in Alberta, Canada. **Prerequisite: 6 months of clean live track record on V1.** Don't market or productize until performance is provable.

---

*Owner: Travis Mitten · Pacific Time · last updated when this commit landed*
