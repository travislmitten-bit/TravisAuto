"""
telegram.py — Minimal Telegram alert sender for TravisAuto.

SETUP (free, 5 minutes):
  1. Open Telegram, message @BotFather → /newbot
  2. Choose a name (e.g. "TravisAutoAlerts") and a username ending in 'bot'
  3. BotFather returns an HTTP API token → paste into .env as TELEGRAM_BOT_TOKEN
  4. Message your new bot at least once (so it can DM you back)
  5. Open https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a browser
  6. Find {"message":{"chat":{"id":<number>...}}} → paste into .env as TELEGRAM_CHAT_ID
  7. Restart the bot — alerts will fire on entry/exit/kill switch/restart/CLARITY window

Behaviour:
  - Best-effort: timeouts/HTTP errors log a warning but never raise
  - No queueing/persistence: if Telegram is down, alerts are dropped
  - Hard-disabled cleanly when token or chat_id is empty
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self._token   = (token   or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID",   "")).strip()
        self._enabled = bool(self._token and self._chat_id)
        if not self._enabled:
            logger.info("Telegram notifier disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, text: str):
        """Fire-and-forget. Never raises. Truncates to 4096 chars (Telegram limit)."""
        if not self._enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                data={
                    "chat_id": self._chat_id,
                    "text":    text[:4096],
                    "disable_web_page_preview": "true",
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
