import time
import hashlib
import hmac
import base64
import urllib.parse
from typing import Dict, List, Optional, Tuple
import requests
import logging

logger = logging.getLogger(__name__)


class KrakenAPI:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.kraken.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TravisAuto/1.0"})

    def _sign(self, url_path: str, data: dict) -> str:
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + post_data).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _public(self, method: str, params: dict = None) -> dict:
        url = f"{self.base_url}/0/public/{method}"
        resp = self.session.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken public error: {data['error']}")
        return data["result"]

    def _private(self, method: str, params: dict = None) -> dict:
        params = params or {}
        params["nonce"] = str(int(time.time() * 1000))
        url_path = f"/0/private/{method}"
        url = self.base_url + url_path
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, params),
        }
        resp = self.session.post(url, data=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken private error: {data['error']}")
        return data["result"]

    # ── Market data ──────────────────────────────────────────────────────────

    def get_ohlcv(self, pair: str, interval: int = 60, since: int = None) -> List[List]:
        """Returns list of [time, open, high, low, close, vwap, volume, count]."""
        params = {"pair": pair, "interval": interval}
        if since:
            params["since"] = since
        result = self._public("OHLC", params)
        key = [k for k in result if k != "last"][0]
        return result[key]

    def get_ticker(self, pair: str) -> dict:
        result = self._public("Ticker", {"pair": pair})
        key = list(result.keys())[0]
        return result[key]

    def get_order_book(self, pair: str, count: int = 10) -> dict:
        result = self._public("Depth", {"pair": pair, "count": count})
        key = list(result.keys())[0]
        return result[key]

    def get_asset_pairs(self) -> dict:
        return self._public("AssetPairs")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> Dict[str, float]:
        result = self._private("Balance")
        return {k: float(v) for k, v in result.items()}

    def get_open_orders(self) -> dict:
        return self._private("OpenOrders")

    def get_trade_balance(self, asset: str = "ZUSD") -> dict:
        return self._private("TradeBalance", {"asset": asset})

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(self, pair: str, side: str, volume: float, dry_run: bool = True) -> dict:
        params = {
            "pair": pair,
            "type": side,           # "buy" or "sell"
            "ordertype": "market",
            "volume": str(round(volume, 8)),
        }
        if dry_run:
            params["validate"] = "true"
        return self._private("AddOrder", params)

    def place_limit_order(
        self, pair: str, side: str, volume: float, price: float, dry_run: bool = True
    ) -> dict:
        params = {
            "pair": pair,
            "type": side,
            "ordertype": "limit",
            "price": str(price),
            "volume": str(round(volume, 8)),
        }
        if dry_run:
            params["validate"] = "true"
        return self._private("AddOrder", params)

    def place_stop_loss_limit(
        self,
        pair: str,
        side: str,
        volume: float,
        stop_price: float,
        limit_price: float,
        dry_run: bool = True,
    ) -> dict:
        params = {
            "pair": pair,
            "type": side,
            "ordertype": "stop-loss-limit",
            "price": str(stop_price),
            "price2": str(limit_price),
            "volume": str(round(volume, 8)),
        }
        if dry_run:
            params["validate"] = "true"
        return self._private("AddOrder", params)

    def cancel_order(self, txid: str) -> dict:
        return self._private("CancelOrder", {"txid": txid})

    def cancel_all_orders(self) -> dict:
        return self._private("CancelAll")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_mid_price(self, pair: str) -> float:
        ticker = self.get_ticker(pair)
        bid = float(ticker["b"][0])
        ask = float(ticker["a"][0])
        return (bid + ask) / 2

    def get_min_order_size(self, pair: str) -> Tuple[float, int]:
        """Returns (min_volume, price_decimals)."""
        pairs = self.get_asset_pairs()
        info = pairs.get(pair) or pairs.get(pair.replace("/", ""))
        if not info:
            return 0.0001, 1
        return float(info.get("ordermin", 0.0001)), int(info.get("pair_decimals", 1))
