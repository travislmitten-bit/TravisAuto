"""
data_scanner.py — Free data aggregation scanner for market sentiment.

7 confirmed-free sources (no paid subscription required):
  1. Fear & Greed Index   — api.alternative.me/fng/
  2. Kraken Futures       — futures.kraken.com (funding rates, no auth)
  3. CoinGecko free API   — coingecko.com (24h momentum, no key)
  4. Blockchain.com stats — api.blockchain.info/stats (BTC on-chain)
  5. Mempool.space        — mempool.space/api (BTC fee market activity)
  6. CryptoPanic RSS      — cryptopanic.com/news/rss (news sentiment)
  7. Solscan chaininfo    — public-api.solscan.io/chaininfo (SOL activity)
  8. Taostats API         — api.taostats.io (optional — needs TAOSTATS_API_KEY in .env)

Composite score 0-100:
  > 70  → BULLISH  — size ×1.2, confidence threshold −0.05
  40-70 → NEUTRAL  — standard sizing
  < 40  → BEARISH  — size ×0.7, confidence threshold +0.10

Cache TTL: 4 hours (never hammers free APIs).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Weights (must sum to 1.0) ─────────────────────────────────────────────────

_WEIGHTS: Dict[str, float] = {
    "fear_greed":      0.25,
    "funding_rate":    0.20,
    "momentum":        0.20,
    "on_chain_btc":    0.15,
    "news_sentiment":  0.10,
    "mempool":         0.05,
    "solana_activity": 0.05,
}

_CACHE_TTL = 4 * 3600          # 4 hours
_TIMEOUT   = 10                 # per-request timeout (seconds)
_PAIRS_OF_INTEREST = {"btc", "bitcoin", "sol", "solana", "tao", "bittensor", "link", "chainlink"}

# Keyword sets for CryptoPanic sentiment
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


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScanResult:
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

    @property
    def size_scalar(self) -> float:
        if self.composite > 70:
            return 1.2
        elif self.composite >= 40:
            return 1.0
        return 0.7

    @property
    def confidence_delta(self) -> float:
        """Additive adjustment to confidence threshold. Positive = harder to enter."""
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
    """Aggregates free market data into a single composite sentiment score."""

    def __init__(self, taostats_api_key: str = ""):
        self._taostats_key = taostats_api_key
        self._cache: Optional[ScanResult] = None
        self._cache_ts: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "TravisAuto/1.0 (+https://github.com/travislmitten-bit/TravisAuto)",
            "Accept": "application/json",
        })

    # ── Public interface ──────────────────────────────────────────────────────

    def refresh(self, force: bool = False) -> ScanResult:
        """Return cached result unless expired or force=True."""
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
        sol   = self._fetch_solscan_activity()
        tao   = self._fetch_taostats() if self._taostats_key else None

        scores = {
            "fear_greed":      fg,
            "funding_rate":    fund,
            "momentum":        mom,
            "on_chain_btc":    btc,
            "news_sentiment":  news,
            "mempool":         mpool,
            "solana_activity": sol,
        }

        composite = self._weighted_composite(scores)
        sources_ok = sum(1 for v in scores.values() if v is not None) + (1 if tao is not None else 0)
        sources_total = len(scores) + 1  # +1 for taostats slot

        result = ScanResult(
            composite=composite,
            fear_greed=fg,
            funding_rate=fund,
            momentum=mom,
            on_chain_btc=btc,
            news_sentiment=news,
            mempool=mpool,
            solana_activity=sol,
            taostats=tao,
            sources_ok=sources_ok,
            sources_total=sources_total,
        )

        logger.info(
            "DataScanner: composite=%.1f [%s] | sources=%d/%d | "
            "F&G=%.0f fund=%.0f mom=%.0f btc=%.0f news=%.0f mpool=%.0f sol=%.0f",
            result.composite, result.label, sources_ok, sources_total,
            fg or -1, fund or -1, mom or -1, btc or -1,
            news or -1, mpool or -1, sol or -1,
        )
        return result

    # ── Composite calculation ─────────────────────────────────────────────────

    @staticmethod
    def _weighted_composite(scores: Dict[str, Optional[float]]) -> float:
        """
        Weighted average of available scores, renormalising weights so missing
        sources don't drag the composite down.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for key, score in scores.items():
            if score is not None:
                w = _WEIGHTS.get(key, 0.0)
                weighted_sum += score * w
                total_weight += w
        if total_weight < 1e-9:
            return 50.0  # no data → neutral
        return round(weighted_sum / total_weight, 1)

    # ── Source 1: Fear & Greed ────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> Optional[float]:
        try:
            r = self._session.get(
                "https://api.alternative.me/fng/?limit=1", timeout=_TIMEOUT
            )
            r.raise_for_status()
            val = float(r.json()["data"][0]["value"])
            logger.debug("F&G index: %.0f", val)
            return val  # already 0-100
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return None

    # ── Source 2: Kraken Futures funding rate ─────────────────────────────────

    def _fetch_funding_rate(self) -> Optional[float]:
        """
        Funding rate interpretation:
          Negative → shorts dominant → potential squeeze → bullish (80)
          Near zero → balanced → neutral (50)
          Mildly positive (0–0.05%) → longs leading, healthy trend → 65
          High positive (>0.1%) → overheated longs → bearish (25)
        """
        try:
            r = self._session.get(
                "https://futures.kraken.com/derivatives/api/v3/tickers",
                timeout=_TIMEOUT,
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
            # Map to 0-100
            if rate < -0.001:
                return 82.0
            elif rate < 0:
                return 70.0
            elif rate < 0.0005:
                return 65.0
            elif rate < 0.001:
                return 55.0
            elif rate < 0.002:
                return 40.0
            else:
                return 22.0
        except Exception as e:
            logger.warning("Funding rate fetch failed: %s", e)
            return None

    # ── Source 3: CoinGecko 24h momentum ─────────────────────────────────────

    def _fetch_momentum(self) -> Optional[float]:
        """
        Average 24h price change across BTC, SOL, TAO, LINK.
        Maps to 0-100 sentiment score.
        """
        try:
            r = self._session.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "bitcoin,solana,bittensor,chainlink",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                },
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            changes = []
            for coin_data in data.values():
                chg = coin_data.get("usd_24h_change")
                if chg is not None:
                    changes.append(float(chg))
            if not changes:
                return None
            avg = sum(changes) / len(changes)
            logger.debug("24h avg momentum: %.2f%%", avg)
            # Map % change to 0-100
            if avg > 8:
                return 90.0
            elif avg > 4:
                return 75.0
            elif avg > 1:
                return 62.0
            elif avg > -1:
                return 50.0
            elif avg > -4:
                return 38.0
            elif avg > -8:
                return 25.0
            else:
                return 12.0
        except Exception as e:
            logger.warning("CoinGecko momentum fetch failed: %s", e)
            return None

    # ── Source 4: Blockchain.com on-chain stats ───────────────────────────────

    def _fetch_blockchain_stats(self) -> Optional[float]:
        """
        BTC on-chain health from n_tx (daily tx count) and block timing.
        Normal healthy range: 200k–400k tx/day, 9–11 min/block.
        """
        try:
            r = self._session.get(
                "https://api.blockchain.info/stats", timeout=_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            n_tx   = int(data.get("n_tx", 0))
            mins   = float(data.get("minutes_between_blocks", 10))
            logger.debug("BTC on-chain: n_tx=%d, mins_per_block=%.1f", n_tx, mins)

            # tx score
            if n_tx > 400_000:
                tx_score = 80.0
            elif n_tx > 300_000:
                tx_score = 65.0
            elif n_tx > 200_000:
                tx_score = 50.0
            elif n_tx > 100_000:
                tx_score = 35.0
            else:
                tx_score = 20.0

            # block timing score (10 min = ideal)
            if 9 <= mins <= 11:
                timing_score = 70.0
            elif 7 <= mins <= 13:
                timing_score = 55.0
            else:
                timing_score = 40.0

            return round(0.7 * tx_score + 0.3 * timing_score, 1)
        except Exception as e:
            logger.warning("Blockchain.com stats fetch failed: %s", e)
            return None

    # ── Source 5: CryptoPanic RSS sentiment ───────────────────────────────────

    def _fetch_cryptopanic_sentiment(self) -> Optional[float]:
        """
        Parse CryptoPanic public RSS (no auth needed).
        Uses regex instead of XML parser because the feed occasionally has
        malformed tags that cause ElementTree to abort.
        """
        try:
            r = self._session.get(
                "https://cryptopanic.com/news/rss",
                headers={"Accept": "application/rss+xml, application/xml, text/xml"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            # Extract raw title strings — handles CDATA and malformed tags
            raw_titles: List[str] = re.findall(
                r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                r.text, re.DOTALL | re.IGNORECASE,
            )
            titles = raw_titles[1:]   # skip the feed title itself

            bullish = 0
            bearish = 0
            relevant = 0
            for title in titles[:50]:
                t = title.lower().strip()
                if not any(w in t for w in (
                    "bitcoin", "btc", "solana", "sol", "tao", "bittensor",
                    "chainlink", "link", "crypto", "market",
                )):
                    continue
                relevant += 1
                for w in _BULLISH_WORDS:
                    if w in t:
                        bullish += 1
                for w in _BEARISH_WORDS:
                    if w in t:
                        bearish += 1

            if relevant == 0:
                return 50.0
            total_sentiment = bullish + bearish
            if total_sentiment == 0:
                return 50.0
            bull_ratio = bullish / total_sentiment
            score = round(bull_ratio * 100, 1)
            logger.debug(
                "CryptoPanic: %d relevant | bullish=%d bearish=%d | score=%.1f",
                relevant, bullish, bearish, score,
            )
            return score
        except Exception as e:
            logger.warning("CryptoPanic sentiment fetch failed: %s", e)
            return None

    # ── Source 6: Mempool.space fee market ────────────────────────────────────

    def _fetch_mempool_fees(self) -> Optional[float]:
        """
        BTC fee market activity. Higher fees = more block space demand = active market.
        Uses fastestFee (sat/vB) as activity proxy.
        """
        try:
            r = self._session.get(
                "https://mempool.space/api/v1/fees/recommended", timeout=_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            fee = float(data.get("fastestFee", 0))
            logger.debug("Mempool fastest fee: %.1f sat/vB", fee)
            if fee < 2:
                return 20.0
            elif fee < 5:
                return 35.0
            elif fee < 15:
                return 50.0
            elif fee < 30:
                return 62.0
            elif fee < 80:
                return 72.0
            elif fee < 200:
                return 62.0   # very busy, slight stress signal
            else:
                return 45.0   # congested / panic-driven
        except Exception as e:
            logger.warning("Mempool.space fetch failed: %s", e)
            return None

    # ── Source 7: Solana RPC — network activity (TPS) ─────────────────────────

    def _fetch_solscan_activity(self) -> Optional[float]:
        """
        Solana TPS via the free public Solana RPC (no auth required).
        getRecentPerformanceSamples returns numTransactions / samplePeriodSecs.
        Normal range: 1 500–4 000 TPS.
        """
        try:
            r = self._session.post(
                "https://api.mainnet-beta.solana.com",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getRecentPerformanceSamples",
                    "params": [4],
                },
                headers={"Content-Type": "application/json"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            samples = r.json().get("result", [])
            if not samples:
                return None
            valid = [
                s["numTransactions"] / max(s["samplePeriodSecs"], 1)
                for s in samples
                if "numTransactions" in s and "samplePeriodSecs" in s
            ]
            if not valid:
                return None
            avg_tps = sum(valid) / len(valid)
            logger.debug("Solana avg TPS (RPC): %.1f", avg_tps)
            if avg_tps > 3500:
                return 80.0
            elif avg_tps > 2000:
                return 65.0
            elif avg_tps > 800:
                return 50.0
            elif avg_tps > 200:
                return 38.0
            else:
                return 25.0
        except Exception as e:
            logger.warning("Solana RPC fetch failed: %s", e)
            return None

    # ── Source 8: Taostats (optional) ────────────────────────────────────────

    def _fetch_taostats(self) -> Optional[float]:
        """
        TAO-specific on-chain data. Requires TAOSTATS_API_KEY in .env.
        High staking ratio = strong holder conviction = bullish.
        """
        try:
            r = self._session.get(
                "https://api.taostats.io/api/v1/stats",
                headers={"Authorization": f"Bearer {self._taostats_key}"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            staking_ratio = data.get("staking_ratio") or data.get("stakingRatio")
            if staking_ratio is None:
                return None
            ratio = float(staking_ratio)
            logger.debug("TAO staking ratio: %.3f", ratio)
            # ratio = staked / circulating supply (0–1)
            if ratio > 0.65:
                return 78.0
            elif ratio > 0.50:
                return 62.0
            elif ratio > 0.35:
                return 50.0
            else:
                return 35.0
        except Exception as e:
            logger.debug("Taostats fetch failed (key set=%s): %s", bool(self._taostats_key), e)
            return None


# ── Convenience accessor ──────────────────────────────────────────────────────

def neutral_result() -> ScanResult:
    """Fallback result used before first scan completes."""
    return ScanResult(
        composite=50.0,
        fear_greed=None,
        funding_rate=None,
        momentum=None,
        on_chain_btc=None,
        news_sentiment=None,
        mempool=None,
        solana_activity=None,
        taostats=None,
        sources_ok=0,
        sources_total=8,
    )
