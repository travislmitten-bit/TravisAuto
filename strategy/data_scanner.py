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

Composite score 0-100:
  > 70  → BULLISH  — size ×1.2, confidence −0.05
  40-70 → NEUTRAL  — standard sizing
  < 40  → BEARISH  — size ×0.7, confidence +0.10

Cache TTL: 4 hours.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import requests

# yfinance is optional — cross-asset correlation degrades gracefully without it
try:
    import yfinance as yf
    import pandas as pd
    _YFINANCE_OK = True
except ImportError:
    _YFINANCE_OK = False

logger = logging.getLogger(__name__)

_WEIGHTS: Dict[str, float] = {
    "fear_greed":      0.25,
    "funding_rate":    0.20,
    "momentum":        0.20,
    "on_chain_btc":    0.15,
    "news_sentiment":  0.10,
    "mempool":         0.05,
    "solana_activity": 0.05,
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

# Kraken pair → LunarCrush symbol
_PAIR_TO_LC: Dict[str, str] = {
    "XBTUSD":  "BTC",
    "SOLUSD":  "SOL",
    "TAOUSD":  "TAO",
    "LINKUSD": "LINK",
}


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
    btc_sp_corr:          Optional[float] = None
    btc_dxy_corr:         Optional[float] = None
    btc_gold_corr:        Optional[float] = None
    corr_adjustment:      float = 0.0      # net pts applied to composite
    lunarcrush:           Dict[str, dict] = field(default_factory=dict)
    smart_money_divergence: bool = False

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

    def __init__(self, taostats_api_key: str = "", lunarcrush_api_key: str = ""):
        self._taostats_key    = taostats_api_key
        self._lunarcrush_key  = lunarcrush_api_key
        self._cache: Optional[ScanResult] = None
        self._cache_ts: float = 0.0
        # AltRank history for 24h social momentum detection
        # deque maxlen=8 → ~32h at 4h refresh rate
        self._altrank_log: Dict[str, Deque[Tuple[float, int]]] = {}
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
        news  = self._fetch_cryptopanic_sentiment()
        mpool = self._fetch_mempool_fees()
        sol   = self._fetch_solana_tps()
        tao   = self._fetch_taostats() if self._taostats_key else None

        # Godmode: cross-asset correlation (yfinance)
        btc_sp, btc_dxy, btc_gold = self._fetch_cross_asset_correlation()
        corr_adj, corr_reasons = self._apply_corr_adjustment(btc_sp, btc_dxy)
        if corr_reasons:
            logger.info("Corr adjustment %+.0f pts | %s", corr_adj, " | ".join(corr_reasons))

        # Godmode: LunarCrush social sentiment
        lc_data = self._fetch_lunarcrush()

        # Godmode: smart money divergence
        smd = self._detect_smart_money_divergence()

        scores = {
            "fear_greed":      fg,
            "funding_rate":    fund,
            "momentum":        mom,
            "on_chain_btc":    btc,
            "news_sentiment":  news,
            "mempool":         mpool,
            "solana_activity": sol,
        }

        base = self._weighted_composite(scores)
        composite = round(max(0.0, min(100.0, base + corr_adj)), 1)

        sources_ok    = sum(1 for v in scores.values() if v is not None) + (1 if tao is not None else 0)
        sources_total = len(scores) + 1

        result = ScanResult(
            composite=composite,
            fear_greed=fg, funding_rate=fund, momentum=mom,
            on_chain_btc=btc, news_sentiment=news, mempool=mpool,
            solana_activity=sol, taostats=tao,
            sources_ok=sources_ok, sources_total=sources_total,
            btc_sp_corr=btc_sp, btc_dxy_corr=btc_dxy, btc_gold_corr=btc_gold,
            corr_adjustment=corr_adj,
            lunarcrush=lc_data,
            smart_money_divergence=smd,
        )

        logger.info(
            "DataScanner: composite=%.1f [%s] | sources=%d/%d | "
            "F&G=%.0f fund=%.0f mom=%.0f btc=%.0f news=%.0f mpool=%.0f sol=%.0f | "
            "corr_adj=%+.0f | SP=%.2f DXY=%.2f Gold=%.2f | SMD=%s | LC=%s",
            composite, result.label, sources_ok, sources_total,
            fg or -1, fund or -1, mom or -1, btc or -1,
            news or -1, mpool or -1, sol or -1, corr_adj,
            btc_sp or 0, btc_dxy or 0, btc_gold or 0, smd,
            {p: f"GS={v.get('galaxy_score',0):.0f}" for p, v in lc_data.items()},
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

    @staticmethod
    def _apply_corr_adjustment(
        btc_sp: Optional[float], btc_dxy: Optional[float]
    ) -> Tuple[float, List[str]]:
        adj, reasons = 0.0, []
        if btc_sp is not None and btc_sp > 0.7:
            adj -= 10.0
            reasons.append(f"BTC/S&P r={btc_sp:.2f}>0.70 (macro risk −10)")
        if btc_dxy is not None and btc_dxy < -0.6:
            adj += 10.0
            reasons.append(f"BTC/DXY r={btc_dxy:.2f}<−0.60 (dollar weak +10)")
        return adj, reasons

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
            changes = [float(v["usd_24h_change"]) for v in r.json().values()
                       if v.get("usd_24h_change") is not None]
            if not changes:
                return None
            avg = sum(changes) / len(changes)
            logger.debug("24h avg momentum: %.2f%%", avg)
            if avg > 8:   return 90.0
            elif avg > 4: return 75.0
            elif avg > 1: return 62.0
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

    # ── Source 5: CryptoPanic RSS sentiment ───────────────────────────────────

    def _fetch_cryptopanic_sentiment(self) -> Optional[float]:
        try:
            r = self._session.get(
                "https://cryptopanic.com/news/rss",
                headers={"Accept": "application/rss+xml, text/xml"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            titles = re.findall(
                r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                r.text, re.DOTALL | re.IGNORECASE,
            )[1:]
            bullish = bearish = relevant = 0
            for title in titles[:50]:
                t = title.lower().strip()
                if not any(w in t for w in ("bitcoin", "btc", "solana", "sol",
                           "tao", "bittensor", "chainlink", "link", "crypto", "market")):
                    continue
                relevant += 1
                bullish += sum(1 for w in _BULLISH_WORDS if w in t)
                bearish += sum(1 for w in _BEARISH_WORDS if w in t)
            if relevant == 0 or (bullish + bearish) == 0:
                return 50.0
            score = round(bullish / (bullish + bearish) * 100, 1)
            logger.debug("CryptoPanic: rel=%d bull=%d bear=%d → %.1f", relevant, bullish, bearish, score)
            return score
        except Exception as e:
            logger.warning("CryptoPanic sentiment fetch failed: %s", e)
            return None

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

    # ── Godmode 1: Cross-asset correlation ───────────────────────────────────

    def _fetch_cross_asset_correlation(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Download 90 days of daily returns for BTC, S&P500, DXY, Gold via yfinance.
        Returns (btc_sp_corr, btc_dxy_corr, btc_gold_corr).
        Degrades to (None, None, None) if yfinance not installed or any API failure.
        """
        if not _YFINANCE_OK:
            logger.debug("yfinance not installed — cross-asset correlation skipped")
            return None, None, None
        try:
            def _returns(ticker: str):
                data = yf.download(
                    ticker, period="90d", interval="1d",
                    auto_adjust=True, progress=False, threads=False,
                )
                if data.empty:
                    return None
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.pct_change().dropna()

            btc_r  = _returns("BTC-USD")
            sp_r   = _returns("^GSPC")
            dxy_r  = _returns("DX-Y.NYB")
            gold_r = _returns("GC=F")

            def _corr(a, b) -> Optional[float]:
                if a is None or b is None:
                    return None
                idx = a.index.intersection(b.index)
                if len(idx) < 20:
                    return None
                return float(a.loc[idx].corr(b.loc[idx]))

            btc_sp   = _corr(btc_r, sp_r)
            btc_dxy  = _corr(btc_r, dxy_r)
            btc_gold = _corr(btc_r, gold_r)

            logger.info(
                "Cross-asset correlation | BTC/S&P=%.2f | BTC/DXY=%.2f | BTC/Gold=%.2f",
                btc_sp or 0, btc_dxy or 0, btc_gold or 0,
            )
            return btc_sp, btc_dxy, btc_gold
        except Exception as e:
            logger.warning("Cross-asset correlation fetch failed: %s", e)
            return None, None, None

    # ── Godmode 2: LunarCrush social sentiment ────────────────────────────────

    def _fetch_lunarcrush(self) -> Dict[str, dict]:
        """
        Fetch Galaxy Score and AltRank from LunarCrush public coin list.
        Requires LUNARCRUSH_API_KEY in .env.

        Returns {kraken_pair: {galaxy_score, alt_rank, social_momentum}}.
        social_momentum=True when AltRank improves ≥20 positions over 24h.
        """
        if not self._lunarcrush_key:
            return {}
        try:
            r = self._session.get(
                "https://lunarcrush.com/api4/public/coins/list/v1",
                headers={"Authorization": f"Bearer {self._lunarcrush_key}"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            coin_list = r.json().get("data", [])
            symbol_map = {
                c.get("symbol", "").upper(): c
                for c in coin_list if c.get("symbol")
            }

            result: Dict[str, dict] = {}
            now = time.time()

            for pair, symbol in _PAIR_TO_LC.items():
                coin = symbol_map.get(symbol)
                if not coin:
                    continue

                galaxy_score = float(coin.get("galaxy_score") or 0)
                alt_rank     = int(coin.get("alt_rank") or 9999)

                # 24h AltRank improvement detection
                social_momentum = False
                log = self._altrank_log.setdefault(pair, deque(maxlen=8))

                # Find an entry from 20–28h ago (one 24h window at 4h scan cadence)
                for entry_ts, entry_rank in log:
                    age_h = (now - entry_ts) / 3600
                    if 20 <= age_h <= 28:
                        improvement = entry_rank - alt_rank  # positive = rank got better
                        if improvement >= 20:
                            social_momentum = True
                            logger.warning(
                                "SOCIAL MOMENTUM | %s | AltRank %d → %d "
                                "(+%d positions in ~24h) | +25%% size",
                                pair, entry_rank, alt_rank, improvement,
                            )
                        break

                log.append((now, alt_rank))
                result[pair] = {
                    "galaxy_score":    galaxy_score,
                    "alt_rank":        alt_rank,
                    "social_momentum": social_momentum,
                }

            if result:
                logger.info(
                    "LunarCrush | %s",
                    {p: f"GS={v['galaxy_score']:.0f} AR={v['alt_rank']}"
                     for p, v in result.items()},
                )
            return result
        except Exception as e:
            logger.warning("LunarCrush fetch failed: %s", e)
            return {}

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
        sources_ok=0, sources_total=8,
    )
