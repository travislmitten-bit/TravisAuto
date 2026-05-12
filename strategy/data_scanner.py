"""
data_scanner.py — Free data aggregation scanner for market sentiment.

Core sources (free, no key):
  1. Fear & Greed Index   — api.alternative.me/fng/
  2. Kraken Futures       — futures.kraken.com (funding rates)
  3. CoinGecko free API   — coingecko.com (24h momentum + 7d chart)
  4. Blockchain.com stats — api.blockchain.info/stats
  5. Mempool.space        — mempool.space/api (BTC fee market)
  6. CryptoPanic RSS      — cryptopanic.com/news/rss
  7. Solana public RPC    — api.mainnet-beta.solana.com (TPS)
  8. Taostats             — api.taostats.io (opt — TAOSTATS_API_KEY)

Godmode additions:
  9. Cross-asset correlation — yfinance: BTC vs S&P500, DXY, Gold
     BTC/S&P > 0.7  → composite −10 (macro risk mode)
     BTC/DXY < −0.6 → composite +10 (dollar weakening = BTC bullish)
 10. LunarCrush social sentiment — Galaxy Score, AltRank (LUNARCRUSH_API_KEY)
     GS > 60 → +8 confidence boost for that asset
     AltRank improves 20+ in 24h → SOCIAL MOMENTUM (+25% size)
 11. Smart money divergence — BTC at 7-day high + volume spike → bearish flag
     Effect: tighter trailing stops in main.py
 12. BTC dominance — CoinGecko /global: 24h change drives altcoin headwind/tailwind
     Rising >1%  → ALTCOIN_HEADWIND  (SOL/TAO/LINK sizes −30%)
     Falling >1% → ALTCOIN_TAILWIND  (conf threshold −0.05 for altcoins)
 13. Open interest trend — OKX /rubik/stat/contracts/open-interest-volume (free, no key)
     OI↑ + price↑ → CONFIRMED_MOVE      (+5  composite)
     OI↑ + price↓ → SHORT_SQUEEZE_BUILDING (+8 composite)
     OI↓ + price↓ → CAPITULATION         (+10 composite)
     OI↓ + price↑ → WEAK_MOVE            (−5 composite, tighter conf threshold)

Composite score 0-100:
  > 70  → BULLISH  — size ×1.2, confidence −0.05
  40-70 → NEUTRAL  — standard sizing
  < 40  → BEARISH  — size ×0.7, confidence +0.10

Cache TTL: 4 hours.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import requests

from html.parser import HTMLParser

logger = logging.getLogger(__name__)

_WEIGHTS: Dict[str, float] = {
    "fear_greed":      0.25,
    "funding_rate":    0.20,
    "momentum":        0.20,
    "on_chain_btc":    0.15,
    "mempool":         0.10,
    "solana_activity": 0.10,
}

# Kraken pair → OKX perpetual instrument (for L/S ratio)
_PAIR_TO_OKX: Dict[str, str] = {
    "XBTUSD":  "BTC-USDT-SWAP",
    "SOLUSD":  "SOL-USDT-SWAP",
    "TAOUSD":  "TAO-USDT-SWAP",
    "LINKUSD": "LINK-USDT-SWAP",
}

_CACHE_TTL = 4 * 3600
_TIMEOUT   = 15   # slightly longer to handle yfinance

_BULLISH_WORDS = {
    "surge", "rally", "bull", "pump", "breakout", "adoption", "institutional",
    "buy", "long", "ath", "all-time high", "record", "growth", "upgrade",
    "partnership", "launch", "milestone", "accumulate", "outperform",
}
_BEARISH_WORDS = {
    "crash", "dump", "bear", "drop", "plunge", "fear", "hack", "exploit",
    "ban", "sell", "short", "warning", "regulation", "lawsuit", "loss",
    "liquidation", "contagion", "fraud", "collapse", "risk",
}

# Kraken pair → CoinGecko ID (for community data)
_PAIR_TO_CG: Dict[str, str] = {
    "XBTUSD":  "bitcoin",
    "SOLUSD":  "solana",
    "TAOUSD":  "bittensor",
    "LINKUSD": "chainlink",
}

# BTC community size used to normalise galaxy_score (log10(~14M) ≈ 7.15)
_CG_SOCIAL_LOG_CEIL = 7.15


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    # Core composite sources
    composite:        float
    fear_greed:       Optional[float]
    funding_rate:     Optional[float]
    momentum:         Optional[float]
    on_chain_btc:     Optional[float]
    news_sentiment:   Optional[float]
    mempool:          Optional[float]
    solana_activity:  Optional[float]
    taostats:         Optional[float]
    sources_ok:       int
    sources_total:    int
    timestamp:        float = field(default_factory=time.time)

    # Godmode additions
    lunarcrush:             Dict[str, dict] = field(default_factory=dict)
    smart_money_divergence: bool = False
    sector_rotation:        Dict[str, float] = field(default_factory=dict)

    # ETF flows (farside.co.uk)
    etf_flow_adj:    float = 0.0   # direct composite pts: +8 / -8 / 0
    etf_flow_streak: int   = 0     # positive = inflow streak, negative = outflow streak

    # NUPL cycle (Glassnode)
    nupl:            Optional[float] = None
    nupl_signal:     str = "NEUTRAL"   # OPPORTUNISTIC | CAUTION | NEUTRAL

    # Long/Short ratios (OKX per asset)
    ls_ratios:    Dict[str, float] = field(default_factory=dict)
    ls_size_adjs: Dict[str, float] = field(default_factory=dict)  # pair → 1.25/0.75/1.0
    ls_comp_adj:  float = 0.0

    # BTC dominance
    btc_dominance:      Optional[float] = None   # current %
    btc_dom_24h_change: Optional[float] = None   # pct-point change vs 24h ago
    dom_signal:         str = "NEUTRAL"           # HEADWIND | TAILWIND | NEUTRAL

    # Open interest trend
    oi_usd:          Optional[float] = None  # total OI in USD
    oi_24h_change:   Optional[float] = None  # % change
    oi_signal:       str = "NEUTRAL"          # CONFIRMED_MOVE | SHORT_SQUEEZE_BUILDING
                                               # CAPITULATION | WEAK_MOVE | NEUTRAL
    oi_adjustment:   float = 0.0              # pts applied to composite

    @property
    def size_scalar(self) -> float:
        if self.composite > 70:
            return 1.2
        elif self.composite >= 40:
            return 1.0
        return 0.7

    @property
    def confidence_delta(self) -> float:
        if self.composite > 70:
            return -0.05
        elif self.composite < 40:
            return 0.10
        return 0.0

    @property
    def label(self) -> str:
        if self.composite > 70:
            return "BULLISH"
        elif self.composite >= 40:
            return "NEUTRAL"
        return "BEARISH"


# ── Scanner ───────────────────────────────────────────────────────────────────

class DataScanner:
    """Aggregates free market data into a composite sentiment score."""

    def __init__(
        self,
        taostats_api_key:  str = "",
        lunarcrush_api_key: str = "",
        glassnode_api_key:  str = "",
    ):
        self._taostats_key    = taostats_api_key
        self._lunarcrush_key  = lunarcrush_api_key
        self._glassnode_key   = glassnode_api_key
        self._cache: Optional[ScanResult] = None
        self._cache_ts: float = 0.0
        self._altrank_log: Dict[str, Deque[Tuple[float, int]]] = {}
        self._dom_history: Deque[Tuple[float, float]] = deque(maxlen=8)
        self._last_dom_pct: Optional[float] = None
        self._last_btc_24h_pct: Optional[float] = None
        # GS history for sector rotation (pair → [(ts, gs_val)])
        self._gs_history: Dict[str, Deque[Tuple[float, float]]] = {}
        self._nupl_last_log: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "TravisAuto/1.0",
            "Accept": "application/json",
        })

    # ── Public interface ──────────────────────────────────────────────────────

    def refresh(self, force: bool = False) -> ScanResult:
        if not force and self._cache and (time.time() - self._cache_ts < _CACHE_TTL):
            return self._cache
        result = self._run_scan()
        self._cache = result
        self._cache_ts = time.time()
        return result

    @property
    def cached(self) -> Optional[ScanResult]:
        return self._cache

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def _run_scan(self) -> ScanResult:
        logger.info("DataScanner: running full market scan …")

        fg    = self._fetch_fear_greed()
        fund  = self._fetch_funding_rate()
        mom   = self._fetch_momentum()
        btc   = self._fetch_blockchain_stats()
        etf_adj, etf_streak = self._fetch_etf_flows()
        mpool = self._fetch_mempool_fees()
        sol   = self._fetch_solana_tps()
        tao   = self._fetch_taostats() if self._taostats_key else None

        lc_data = self._fetch_lunarcrush()
        smd     = self._detect_smart_money_divergence()

        # Sector rotation from GS 24h delta
        sector_rotation: Dict[str, float] = {}
        gs_deltas = {
            p: v.get("gs_24h_delta")
            for p, v in lc_data.items()
            if v.get("gs_24h_delta") is not None
        }
        if gs_deltas:
            best = max(gs_deltas, key=lambda p: gs_deltas[p])  # type: ignore[arg-type]
            for pair, delta in gs_deltas.items():
                if delta > 2 and pair == best:
                    sector_rotation[pair] = 1.4
                elif delta < -2:
                    sector_rotation[pair] = 0.7
                else:
                    sector_rotation[pair] = 1.0
            non_neutral = {p: v for p, v in sector_rotation.items() if v != 1.0}
            if non_neutral:
                logger.info(
                    "SECTOR_ROTATION | %s",
                    {p: f"{v:.1f}x" for p, v in sector_rotation.items()},
                )

        nupl, nupl_signal = self._fetch_nupl()
        ls_ratios, ls_size_adjs, ls_comp_adj = self._fetch_long_short_ratios()

        # 60s pause after social calls to clear CoinGecko rate limit
        time.sleep(60)
        dom_current, dom_24h_change, dom_signal = self._fetch_btc_dominance()
        oi_usd, oi_24h_change, oi_signal, oi_adj = self._fetch_open_interest()

        scores = {
            "fear_greed":      fg,
            "funding_rate":    fund,
            "momentum":        mom,
            "on_chain_btc":    btc,
            "mempool":         mpool,
            "solana_activity": sol,
        }

        base      = self._weighted_composite(scores)
        composite = round(max(0.0, min(100.0, base + etf_adj + oi_adj + ls_comp_adj)), 1)

        sources_ok    = sum(1 for v in scores.values() if v is not None) + (1 if tao is not None else 0)
        sources_total = len(scores) + 1

        result = ScanResult(
            composite=composite,
            fear_greed=fg, funding_rate=fund, momentum=mom,
            on_chain_btc=btc, news_sentiment=None, mempool=mpool,
            solana_activity=sol, taostats=tao,
            sources_ok=sources_ok, sources_total=sources_total,
            lunarcrush=lc_data,
            smart_money_divergence=smd,
            sector_rotation=sector_rotation,
            etf_flow_adj=etf_adj,
            etf_flow_streak=etf_streak,
            nupl=nupl,
            nupl_signal=nupl_signal,
            ls_ratios=ls_ratios,
            ls_size_adjs=ls_size_adjs,
            ls_comp_adj=ls_comp_adj,
            btc_dominance=dom_current,
            btc_dom_24h_change=dom_24h_change,
            dom_signal=dom_signal,
            oi_usd=oi_usd,
            oi_24h_change=oi_24h_change,
            oi_signal=oi_signal,
            oi_adjustment=oi_adj,
        )

        logger.info(
            "DataScanner: composite=%.1f [%s] | sources=%d/%d | "
            "F&G=%.0f fund=%.0f mom=%.0f btc=%.0f mpool=%.0f sol=%.0f | "
            "ETF=%+.0fpts(streak=%+d) | NUPL=%s(%s) | SMD=%s | LC=%s | "
            "DOM=%s (%s) | OI=%s [%s %+.0f pts] | L/S=%s",
            composite, result.label, sources_ok, sources_total,
            fg or -1, fund or -1, mom or -1, btc or -1, mpool or -1, sol or -1,
            etf_adj, etf_streak,
            f"{nupl:.3f}" if nupl is not None else "N/A", nupl_signal,
            smd,
            {p: f"GS={v.get('galaxy_score', 0):.0f}" for p, v in lc_data.items()},
            f"{dom_current:.1f}%" if dom_current is not None else "N/A",
            f"{dom_signal} {dom_24h_change:+.2f}pp" if dom_24h_change is not None else dom_signal,
            (
                f"${oi_usd/1e9:.1f}B "
                + ("RISING" if oi_24h_change and oi_24h_change > 0 else "FALLING")
                + f" {oi_24h_change:+.1f}%"
                if oi_usd and oi_24h_change is not None else "N/A"
            ),
            oi_signal, oi_adj,
            {p: f"{v:.2f}" for p, v in ls_ratios.items()} if ls_ratios else "N/A",
        )
        return result

    # ── Composite ─────────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_composite(scores: Dict[str, Optional[float]]) -> float:
        total_w, weighted_sum = 0.0, 0.0
        for key, score in scores.items():
            if score is not None:
                w = _WEIGHTS.get(key, 0.0)
                weighted_sum += score * w
                total_w += w
        return round(weighted_sum / total_w, 1) if total_w > 1e-9 else 50.0

    # ── Source 1: Fear & Greed ────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> Optional[float]:
        try:
            r = self._session.get("https://api.alternative.me/fng/?limit=1", timeout=_TIMEOUT)
            r.raise_for_status()
            val = float(r.json()["data"][0]["value"])
            logger.debug("F&G: %.0f", val)
            return val
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return None

    # ── Source 2: Kraken Futures funding rate ─────────────────────────────────

    def _fetch_funding_rate(self) -> Optional[float]:
        try:
            r = self._session.get(
                "https://futures.kraken.com/derivatives/api/v3/tickers", timeout=_TIMEOUT
            )
            r.raise_for_status()
            tickers = r.json().get("tickers", [])
            rate = None
            for t in tickers:
                if t.get("symbol") == "PF_XBTUSD":
                    rate = t.get("fundingRate")
                    break
            if rate is None:
                return None
            rate = float(rate)
            logger.debug("BTC funding rate: %.6f", rate)
            if rate < -0.001:  return 82.0
            elif rate < 0:     return 70.0
            elif rate < 0.0005: return 65.0
            elif rate < 0.001: return 55.0
            elif rate < 0.002: return 40.0
            else:              return 22.0
        except Exception as e:
            logger.warning("Funding rate fetch failed: %s", e)
            return None

    # ── Source 3: CoinGecko 24h momentum ─────────────────────────────────────

    def _fetch_momentum(self) -> Optional[float]:
        try:
            r = self._session.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin,solana,bittensor,chainlink",
                        "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            payload = r.json()
            # Store BTC 24h pct for OI trend comparison
            btc_data = payload.get("bitcoin", {})
            if btc_data.get("usd_24h_change") is not None:
                self._last_btc_24h_pct = float(btc_data["usd_24h_change"])
            changes = [float(v["usd_24h_change"]) for v in payload.values()
                       if v.get("usd_24h_change") is not None]
            if not changes:
                return None
            avg = sum(changes) / len(changes)
            logger.debug("24h avg momentum: %.2f%%", avg)
            if avg > 8:    return 90.0
            elif avg > 4:  return 75.0
            elif avg > 1:  return 62.0
            elif avg > -1: return 50.0
            elif avg > -4: return 38.0
            elif avg > -8: return 25.0
            else:          return 12.0
        except Exception as e:
            logger.warning("CoinGecko momentum fetch failed: %s", e)
            return None

    # ── Source 4: Blockchain.com on-chain stats ───────────────────────────────

    def _fetch_blockchain_stats(self) -> Optional[float]:
        try:
            r = self._session.get("https://api.blockchain.info/stats", timeout=_TIMEOUT)
            r.raise_for_status()
            data  = r.json()
            n_tx  = int(data.get("n_tx", 0))
            mins  = float(data.get("minutes_between_blocks", 10))
            logger.debug("BTC on-chain: n_tx=%d mins/block=%.1f", n_tx, mins)
            tx_score = 80.0 if n_tx > 400_000 else 65.0 if n_tx > 300_000 \
                       else 50.0 if n_tx > 200_000 else 35.0 if n_tx > 100_000 else 20.0
            timing   = 70.0 if 9 <= mins <= 11 else 55.0 if 7 <= mins <= 13 else 40.0
            return round(0.7 * tx_score + 0.3 * timing, 1)
        except Exception as e:
            logger.warning("Blockchain.com stats fetch failed: %s", e)
            return None

    # ── Source 5: Bitcoin ETF flows (farside.co.uk) ──────────────────────────

    def _fetch_etf_flows(self) -> Tuple[float, int]:
        """
        Scrapes farside.co.uk daily ETF flow totals.
        5+ consecutive positive days → +8 composite.
        3+ consecutive negative days → −8 composite.
        Returns (composite_adj, streak) where streak sign = direction.
        """
        try:
            r = self._session.get(
                "https://farside.co.uk/bitcoin-etf-flow-all-data-table/",
                timeout=_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                },
            )
            r.raise_for_status()

            class _RowParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows: List[List[str]] = []
                    self._row: List[str] = []
                    self._cell = ""
                    self._in_cell = False
                    self._in_body = False

                def handle_starttag(self, tag, attrs):
                    if tag == "tbody":
                        self._in_body = True
                    elif tag == "tr" and self._in_body:
                        self._row = []
                    elif tag in ("td", "th") and self._in_body:
                        self._in_cell = True
                        self._cell = ""

                def handle_endtag(self, tag):
                    if tag == "tbody":
                        self._in_body = False
                    elif tag == "tr" and self._in_body and self._row:
                        self.rows.append(self._row[:])
                        self._row = []
                    elif tag in ("td", "th") and self._in_cell:
                        self._row.append(self._cell.strip())
                        self._in_cell = False

                def handle_data(self, data):
                    if self._in_cell:
                        self._cell += data

            parser = _RowParser()
            parser.feed(r.text)

            totals: List[float] = []
            for row in parser.rows:
                if not row:
                    continue
                raw = row[-1].replace(",", "").replace("$", "").strip()
                if raw in ("-", "", "Total", "Totals"):
                    continue
                try:
                    totals.append(float(raw))
                except ValueError:
                    pass

            if len(totals) < 3:
                logger.debug("ETF flows: insufficient data (%d rows)", len(totals))
                return 0.0, 0

            recent = totals[-10:]
            sign   = 1 if recent[-1] > 0 else -1
            streak = 0
            for v in reversed(recent):
                if (v > 0 and sign > 0) or (v < 0 and sign < 0):
                    streak += 1
                else:
                    break
            streak *= sign

            if streak >= 5:
                adj = 8.0
                logger.info("ETF FLOWS | +%d consecutive positive days → +8 composite", abs(streak))
            elif streak <= -3:
                adj = -8.0
                logger.warning(
                    "ETF FLOWS | %d consecutive negative days → −8 composite | "
                    "tighten stops 20%%",
                    abs(streak),
                )
            else:
                adj = 0.0
                logger.debug("ETF FLOWS | streak=%+d | NEUTRAL", streak)

            return adj, streak
        except Exception as e:
            logger.warning("ETF flow primary (farside) failed: %s — trying Yahoo fallback", e)
            return self._fetch_etf_flows_yahoo()

    def _fetch_etf_flows_yahoo(self) -> Tuple[float, int]:
        """
        Fallback ETF flow proxy using Yahoo Finance v8 chart endpoint.
        Aggregates signed daily dollar-flow across IBIT, FBTC, BITB, ARKB
        (~80% of spot BTC ETF AUM). Net flow proxy = sign(close-open) × volume × close.
        Returns (composite_adj, streak) using the same +8/-8 thresholds as farside.
        """
        tickers = ["IBIT", "FBTC", "BITB", "ARKB"]
        daily_totals: dict[int, float] = {}
        for ticker in tickers:
            try:
                r = self._session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                    params={"interval": "1d", "range": "30d"},
                    timeout=_TIMEOUT,
                    headers={"User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    )},
                )
                r.raise_for_status()
                payload = r.json().get("chart", {}).get("result", [])
                if not payload:
                    continue
                ts_arr  = payload[0].get("timestamp", []) or []
                quote   = payload[0].get("indicators", {}).get("quote", [{}])[0]
                opens   = quote.get("open",   []) or []
                closes  = quote.get("close",  []) or []
                vols    = quote.get("volume", []) or []
                for ts, o, c, v in zip(ts_arr, opens, closes, vols):
                    if None in (o, c, v) or v == 0:
                        continue
                    # Bucket by UTC date
                    day_key = ts - (ts % 86400)
                    signed_flow = (1.0 if c > o else -1.0 if c < o else 0.0) * float(c) * float(v)
                    daily_totals[day_key] = daily_totals.get(day_key, 0.0) + signed_flow
            except Exception as e:
                logger.debug("Yahoo ETF fetch failed for %s: %s", ticker, e)

        if len(daily_totals) < 3:
            logger.warning("ETF flows fallback (Yahoo): insufficient data (%d days)", len(daily_totals))
            return 0.0, 0

        ordered = sorted(daily_totals.items())            # oldest → newest
        flows   = [f for _, f in ordered[-10:]]           # last 10 trading days

        sign = 1 if flows[-1] > 0 else -1
        streak = 0
        for v in reversed(flows):
            if (v > 0 and sign > 0) or (v < 0 and sign < 0):
                streak += 1
            else:
                break
        streak *= sign

        if streak >= 5:
            adj = 8.0
            logger.info(
                "ETF FLOWS (Yahoo proxy) | +%d consecutive inflow days → +8 composite",
                abs(streak),
            )
        elif streak <= -3:
            adj = -8.0
            logger.warning(
                "ETF FLOWS (Yahoo proxy) | %d consecutive outflow days → −8 composite | "
                "tighten stops 20%%",
                abs(streak),
            )
        else:
            adj = 0.0
            logger.info(
                "ETF FLOWS (Yahoo proxy) | streak=%+d (%d-day window) | NEUTRAL",
                streak, len(flows),
            )
        return adj, streak

    # ── Source 6: Mempool.space fee market ────────────────────────────────────

    def _fetch_mempool_fees(self) -> Optional[float]:
        try:
            r = self._session.get("https://mempool.space/api/v1/fees/recommended", timeout=_TIMEOUT)
            r.raise_for_status()
            fee = float(r.json().get("fastestFee", 0))
            logger.debug("Mempool fastest fee: %.1f sat/vB", fee)
            if fee < 2:    return 20.0
            elif fee < 5:  return 35.0
            elif fee < 15: return 50.0
            elif fee < 30: return 62.0
            elif fee < 80: return 72.0
            elif fee < 200: return 62.0
            else:           return 45.0
        except Exception as e:
            logger.warning("Mempool.space fetch failed: %s", e)
            return None

    # ── Source 7: Solana public RPC TPS ──────────────────────────────────────

    def _fetch_solana_tps(self) -> Optional[float]:
        try:
            r = self._session.post(
                "https://api.mainnet-beta.solana.com",
                json={"jsonrpc": "2.0", "id": 1,
                      "method": "getRecentPerformanceSamples", "params": [4]},
                headers={"Content-Type": "application/json"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            samples = r.json().get("result", [])
            if not samples:
                return None
            valid = [s["numTransactions"] / max(s["samplePeriodSecs"], 1)
                     for s in samples if "numTransactions" in s and "samplePeriodSecs" in s]
            if not valid:
                return None
            tps = sum(valid) / len(valid)
            logger.debug("Solana avg TPS: %.1f", tps)
            if tps > 3500: return 80.0
            elif tps > 2000: return 65.0
            elif tps > 800: return 50.0
            elif tps > 200: return 38.0
            else:           return 25.0
        except Exception as e:
            logger.warning("Solana RPC fetch failed: %s", e)
            return None

    # ── Source 8: Taostats (optional) ────────────────────────────────────────

    def _fetch_taostats(self) -> Optional[float]:
        try:
            r = self._session.get(
                "https://api.taostats.io/api/v1/stats",
                headers={"Authorization": f"Bearer {self._taostats_key}"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            ratio = data.get("staking_ratio") or data.get("stakingRatio")
            if ratio is None:
                return None
            ratio = float(ratio)
            logger.debug("TAO staking ratio: %.3f", ratio)
            if ratio > 0.65: return 78.0
            elif ratio > 0.50: return 62.0
            elif ratio > 0.35: return 50.0
            else:              return 35.0
        except Exception as e:
            logger.debug("Taostats fetch failed: %s", e)
            return None

    # ── Opt 1: BTC dominance ─────────────────────────────────────────────────

    def _fetch_btc_dominance(
        self,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        CoinGecko /global → market_cap_percentage.btc.
        Retries up to 3 times with exponential backoff on 429.
        Returns (current_dom_pct, 24h_change_pp, signal).
        Signal: HEADWIND if +>1pp in 24h, TAILWIND if −>1pp.
        """
        try:
            dom: Optional[float] = None
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    r = self._session.get(
                        "https://api.coingecko.com/api/v3/global", timeout=_TIMEOUT
                    )
                    if r.status_code == 429:
                        wait = 15 * (2 ** attempt)   # 15s, 30s, 60s
                        logger.debug(
                            "CoinGecko 429 on dominance (attempt %d/3), backoff %ds",
                            attempt + 1, wait,
                        )
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    raw_dom = (
                        r.json().get("data", {})
                                .get("market_cap_percentage", {})
                                .get("btc")
                    )
                    if raw_dom is not None:
                        dom = float(raw_dom)
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(10 * (2 ** attempt))
            if dom is None:
                raise ValueError(
                    f"btc dominance field missing from response"
                ) from last_exc
            dom = float(raw_dom)
            self._last_dom_pct = dom   # cache for rate-limit fallback
            now = time.time()

            # Find entry from 20-28h ago before appending current
            dom_24h_change: Optional[float] = None
            for ts, past_dom in self._dom_history:
                age_h = (now - ts) / 3600
                if 20 <= age_h <= 28:
                    dom_24h_change = round(dom - past_dom, 3)
                    break

            self._dom_history.append((now, dom))

            signal = "NEUTRAL"
            if dom_24h_change is not None:
                if dom_24h_change > 1.0:
                    signal = "HEADWIND"
                    logger.warning(
                        "BTC DOMINANCE | %.1f%% | +%.2fpp in 24h → ALTCOIN_HEADWIND "
                        "| SOL/TAO/LINK sizes −30%%",
                        dom, dom_24h_change,
                    )
                elif dom_24h_change < -1.0:
                    signal = "TAILWIND"
                    logger.info(
                        "BTC DOMINANCE | %.1f%% | %.2fpp in 24h → ALTCOIN_TAILWIND "
                        "| conf threshold −0.05 for altcoins",
                        dom, dom_24h_change,
                    )
                else:
                    logger.debug(
                        "BTC dominance: %.1f%% | 24h change: %+.2fpp | NEUTRAL",
                        dom, dom_24h_change,
                    )
            else:
                logger.debug("BTC dominance: %.1f%% | 24h change: accumulating", dom)

            return dom, dom_24h_change, signal
        except Exception as e:
            logger.warning("BTC dominance fetch failed: %s", e)
            # Return cached value if available (preserves 24h history tracking)
            if self._last_dom_pct is not None:
                logger.debug("Using cached dominance %.1f%%", self._last_dom_pct)
                return self._last_dom_pct, None, "NEUTRAL"
            return None, None, "NEUTRAL"

    # ── Opt 2: Open interest trend ────────────────────────────────────────────

    def _fetch_open_interest(
        self,
    ) -> Tuple[Optional[float], Optional[float], str, float]:
        """
        OKX /rubik/stat/contracts/open-interest-volume — free, no key required.
        25 × 1h buckets (~24h window). Compares newest vs 24h-ago entry.
        Returns (oi_usd, oi_24h_change_pct, signal, composite_adjustment).
        Rows format: [timestamp_ms, oi_usd, volume_usd] — newest first.
        """
        try:
            r = self._session.get(
                "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
                params={"ccy": "BTC", "period": "1H", "limit": "25"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            rows = r.json().get("data", [])
            if len(rows) < 2:
                return None, None, "NEUTRAL", 0.0

            current_oi = float(rows[0][1])
            old_oi     = float(rows[min(23, len(rows) - 1)][1])  # ~24h ago
            oi_change  = (current_oi - old_oi) / (old_oi + 1e-9) * 100.0

            btc_pct = self._last_btc_24h_pct  # set by _fetch_momentum()

            oi_rising    = oi_change  >  1.0
            oi_falling   = oi_change  < -1.0
            price_rising  = btc_pct is not None and btc_pct >  0.5
            price_falling = btc_pct is not None and btc_pct < -0.5

            if oi_rising and price_rising:
                signal, adj = "CONFIRMED_MOVE", 5.0
                logger.info(
                    "OI SIGNAL | CONFIRMED_MOVE | OI %+.1f%% + price %+.1f%% "
                    "| real new money → +5 composite",
                    oi_change, btc_pct or 0,
                )
            elif oi_rising and price_falling:
                signal, adj = "SHORT_SQUEEZE_BUILDING", 8.0
                logger.warning(
                    "OI SIGNAL | SHORT_SQUEEZE_BUILDING | OI %+.1f%% + price %+.1f%% "
                    "| shorts accumulating → +8 composite",
                    oi_change, btc_pct or 0,
                )
            elif oi_falling and price_falling:
                signal, adj = "CAPITULATION", 10.0
                logger.warning(
                    "OI SIGNAL | CAPITULATION | OI %.1f%% + price %+.1f%% "
                    "| forced selling ending → +10 composite",
                    oi_change, btc_pct or 0,
                )
            elif oi_falling and price_rising:
                signal, adj = "WEAK_MOVE", -5.0
                logger.warning(
                    "OI SIGNAL | WEAK_MOVE | OI %.1f%% + price %+.1f%% "
                    "| leverage unwinding not real buying → −5 composite",
                    oi_change, btc_pct or 0,
                )
            else:
                signal, adj = "NEUTRAL", 0.0
                logger.debug(
                    "OI signal NEUTRAL | OI %+.1f%% | BTC %s",
                    oi_change, f"{btc_pct:+.1f}%" if btc_pct is not None else "N/A",
                )

            return current_oi, round(oi_change, 2), signal, adj
        except Exception as e:
            logger.warning("Open interest fetch failed: %s", e)
            return None, None, "NEUTRAL", 0.0

    # ── Godmode 1: NUPL Cycle Indicator (Glassnode) ──────────────────────────

    def _fetch_nupl(self) -> Tuple[Optional[float], str]:
        """
        Net Unrealised Profit/Loss from Glassnode free API.
        < 0.25 → OPPORTUNISTIC: conf threshold −0.05
        > 0.75 → CAUTION: stops 0.5×, require 90% conf
        Logged at most once per 24h.
        """
        if not self._glassnode_key:
            return None, "NEUTRAL"
        try:
            r = self._session.get(
                "https://api.glassnode.com/v1/metrics/indicators/nupl",
                params={"a": "BTC", "api_key": self._glassnode_key},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                return None, "NEUTRAL"
            nupl = float(data[-1]["v"])
            now  = time.time()

            if nupl > 0.75:
                signal = "CAUTION"
                if now - self._nupl_last_log >= 86400:
                    logger.warning(
                        "NUPL | %.3f > 0.75 → CAUTION | stops 0.5×, require 90%% conf",
                        nupl,
                    )
                    self._nupl_last_log = now
            elif nupl < 0.25:
                signal = "OPPORTUNISTIC"
                if now - self._nupl_last_log >= 86400:
                    logger.info(
                        "NUPL | %.3f < 0.25 → OPPORTUNISTIC | conf threshold −0.05",
                        nupl,
                    )
                    self._nupl_last_log = now
            else:
                signal = "NEUTRAL"

            return nupl, signal
        except Exception as e:
            logger.debug("NUPL fetch failed: %s", e)
            return None, "NEUTRAL"

    # ── Godmode 2: Long/Short Ratio (OKX per asset) ───────────────────────────

    def _fetch_long_short_ratios(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        OKX account L/S ratio per asset.
        < 0.80 → sentiment extreme short → +10 composite, +25% size
        > 2.00 → overleveraged longs → −10 composite, −25% size (stops tightened in main.py)
        Returns (ratios, size_adjs, composite_adj).
        """
        ratios:    Dict[str, float] = {}
        size_adjs: Dict[str, float] = {}
        comp_adj = 0.0

        for pair, inst_id in _PAIR_TO_OKX.items():
            try:
                r = self._session.get(
                    "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
                    params={"instId": inst_id, "period": "4H", "limit": "2"},
                    timeout=_TIMEOUT,
                )
                r.raise_for_status()
                payload = r.json()
                rows = payload.get("data", [])
                if not rows or len(rows[0]) < 3:
                    logger.info(
                        "L/S RATIO | %s (%s) | OKX returned no data | code=%s msg=%s",
                        pair, inst_id, payload.get("code"), payload.get("msg"),
                    )
                    continue
                long_pct  = float(rows[0][1])
                short_pct = float(rows[0][2])
                ls = long_pct / max(short_pct, 1e-9)
                ratios[pair] = round(ls, 3)

                if ls < 0.80:
                    size_adjs[pair] = 1.25
                    comp_adj += 2.5
                    logger.info(
                        "L/S RATIO | %s | %.2f < 0.80 → extreme short sentiment | +25%% size",
                        pair, ls,
                    )
                elif ls > 2.00:
                    size_adjs[pair] = 0.75
                    comp_adj -= 2.5
                    logger.warning(
                        "L/S RATIO | %s | %.2f > 2.00 → overleveraged longs | −25%% size",
                        pair, ls,
                    )
                else:
                    size_adjs[pair] = 1.0
                    logger.debug("L/S RATIO | %s | %.2f | NEUTRAL", pair, ls)
            except Exception as e:
                logger.warning("L/S RATIO | %s (%s) | OKX fetch failed: %s", pair, inst_id, e)

        comp_adj = max(-10.0, min(10.0, comp_adj))
        if ratios:
            logger.info("L/S Ratios | %s", {p: f"{v:.2f}" for p, v in ratios.items()})
        return ratios, size_adjs, comp_adj

    # ── Godmode 3: Social sentiment via CoinGecko community data ─────────────

    # Static baselines — BTC dominates, TAO is niche; used when live data unavailable.
    # GS > 60 triggers confidence boost in main.py.
    _STATIC_GS: Dict[str, float] = {
        "XBTUSD":  95.0,
        "SOLUSD":  82.0,
        "LINKUSD": 72.0,
        "TAOUSD":  45.0,   # below 60 → no automatic confidence boost
    }

    def _fetch_lunarcrush(self) -> Dict[str, dict]:
        """
        Social sentiment using CoinGecko community data (free, no API key).
        Static GS baselines are used when the API is rate-limited or returns null.
        1-second delay between coin calls prevents 429 rate-limit errors.

        Output format: {pair: {galaxy_score, alt_rank, social_momentum}}.
        """
        result: Dict[str, dict] = {}
        now = time.time()

        for i, (pair, cg_id) in enumerate(_PAIR_TO_CG.items()):
            if i > 0:
                time.sleep(1.2)   # CoinGecko free tier: ≤1 call/sec burst limit

            # Determine galaxy_score — try live, fall back to static
            galaxy_score = self._STATIC_GS.get(pair, 50.0)
            comm_size    = 0

            try:
                r = self._session.get(
                    f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                    params={
                        "localization":   "false",
                        "tickers":        "false",
                        "market_data":    "false",
                        "community_data": "true",
                        "developer_data": "false",
                        "sparkline":      "false",
                    },
                    timeout=_TIMEOUT,
                )
                r.raise_for_status()
                comm     = r.json().get("community_data", {}) or {}
                twitter  = int(comm.get("twitter_followers")  or 0)
                reddit   = int(comm.get("reddit_subscribers") or 0)
                comm_size = twitter + reddit

                if comm_size > 0:
                    log_size     = math.log10(comm_size)
                    galaxy_score = round(min(100.0, log_size / _CG_SOCIAL_LOG_CEIL * 100), 1)
                    logger.debug(
                        "Social | %s | twitter=%d reddit=%d | GS=%.0f",
                        pair, twitter, reddit, galaxy_score,
                    )
                else:
                    logger.debug("Social | %s | community_data null — using static GS=%.0f",
                                 pair, galaxy_score)
            except Exception as e:
                logger.debug("Social fetch failed for %s: %s — using static GS=%.0f",
                             pair, e, galaxy_score)

            alt_rank = max(1, round(100 - galaxy_score))

            # Social momentum: ≥5% community growth vs ~24h ago
            social_momentum = False
            log = self._altrank_log.setdefault(pair, deque(maxlen=8))
            if comm_size > 0:
                for entry_ts, entry_size in log:
                    age_h = (now - entry_ts) / 3600
                    if 20 <= age_h <= 28:
                        if entry_size > 0:
                            growth = (comm_size - entry_size) / entry_size
                            if growth >= 0.05:
                                social_momentum = True
                                logger.warning(
                                    "SOCIAL MOMENTUM | %s | community +%.1f%% in ~24h "
                                    "| +25%% size",
                                    pair, growth * 100,
                                )
                        break
                log.append((now, comm_size))

            # GS 24h delta for sector rotation
            gs_24h_delta: Optional[float] = None
            gs_log = self._gs_history.setdefault(pair, deque(maxlen=8))
            for entry_ts, entry_gs in gs_log:
                age_h = (now - entry_ts) / 3600
                if 20 <= age_h <= 28:
                    gs_24h_delta = galaxy_score - entry_gs
                    break
            gs_log.append((now, galaxy_score))

            result[pair] = {
                "galaxy_score":    galaxy_score,
                "alt_rank":        alt_rank,
                "social_momentum": social_momentum,
                "gs_24h_delta":    gs_24h_delta,
            }

        logger.info(
            "Social sentiment | %s",
            {p: f"GS={v['galaxy_score']:.0f}" for p, v in result.items()},
        )
        return result

    # ── Godmode 3: Smart money divergence ────────────────────────────────────

    def _detect_smart_money_divergence(self) -> bool:
        """
        Detect distribution signal: BTC price at 7-day high while
        on-chain volume is ≥30% above the 7-day average.

        Data source: CoinGecko 7-day market chart (prices + total_volumes).
        This is the closest free proxy for exchange inflow pressure.
        """
        try:
            r = self._session.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": "7"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data    = r.json()
            prices  = [p[1] for p in data.get("prices", [])]
            volumes = [v[1] for v in data.get("total_volumes", [])]

            if len(prices) < 7 or len(volumes) < 7:
                return False

            current_price = prices[-1]
            high_7d       = max(prices)
            near_high     = current_price >= high_7d * 0.995   # within 0.5% of 7d high

            if not near_high:
                return False

            avg_vol = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
            cur_vol = volumes[-1]
            elevated = cur_vol > avg_vol * 1.30   # 30%+ above 7d average

            if elevated:
                logger.warning(
                    "SMART MONEY DIVERGENCE | BTC near 7d high $%.0f "
                    "| volume %.2fx avg → tightening all trailing stops",
                    current_price, cur_vol / (avg_vol + 1e-9),
                )
                return True
            return False
        except Exception as e:
            logger.debug("Smart money divergence check failed: %s", e)
            return False


# ── Fallback ──────────────────────────────────────────────────────────────────

def neutral_result() -> ScanResult:
    """Neutral placeholder used before the first scan completes."""
    return ScanResult(
        composite=50.0,
        fear_greed=None, funding_rate=None, momentum=None,
        on_chain_btc=None, news_sentiment=None, mempool=None,
        solana_activity=None, taostats=None,
        sources_ok=0, sources_total=6,
        btc_dominance=None, btc_dom_24h_change=None, dom_signal="NEUTRAL",
        oi_usd=None, oi_24h_change=None, oi_signal="NEUTRAL", oi_adjustment=0.0,
    )
