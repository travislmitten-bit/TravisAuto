import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


@dataclass
class KrakenConfig:
    api_key: str = os.getenv("KRAKEN_API_KEY", "")
    api_secret: str = os.getenv("KRAKEN_API_SECRET", "")
    base_url: str = "https://api.kraken.com"


@dataclass
class TrendlineConfig:
    lookback_candles: int = 100
    min_touches: int = 2
    swing_window: int = 5
    breakout_threshold: float = 0.002   # 0.2% beyond line to confirm break
    touch_tolerance: float = 0.003      # 0.3% tolerance to count as a touch
    min_slope_angle: float = 5.0        # degrees, filters near-flat lines


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01    # 1% of account per trade
    max_open_trades: int = 3
    default_rr_ratio: float = 2.0       # reward:risk
    trailing_stop: bool = True
    trailing_stop_atr_mult: float = 1.5
    max_daily_loss: float = 0.03        # 3% daily drawdown halt


@dataclass
class ScannerConfig:
    taostats_api_key:   str = os.getenv("TAOSTATS_API_KEY", "")
    lunarcrush_api_key: str = os.getenv("LUNARCRUSH_API_KEY", "")
    glassnode_api_key:  str = os.getenv("GLASSNODE_API_KEY", "")
    scan_interval: int = 4 * 3600   # seconds; matches cache TTL


@dataclass
class BotConfig:
    pairs: List[str] = field(default_factory=lambda: [
        p.strip() for p in
        os.getenv("PAIRS", "XBTUSD,SOLUSD,TAOUSD,LINKUSD").split(",")
        if p.strip()
    ])
    interval: int = int(os.getenv("INTERVAL", "240"))
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no")
    poll_seconds: int = 30
    log_level: str = "INFO"
    kraken: KrakenConfig = field(default_factory=KrakenConfig)
    trendline: TrendlineConfig = field(default_factory=TrendlineConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)


CONFIG = BotConfig()
