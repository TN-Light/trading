# ============================================================================
# PROMETHEUS — Interface: Telegram Alert Bot
# ============================================================================
"""
Telegram bot for mobile trading alerts.
Sends signals, P&L updates, risk warnings to your phone.
"""

import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime

from prometheus.utils.logger import logger


class TelegramBot:
    """
    Telegram notification bot for PROMETHEUS.

    Setup:
    1. Message @BotFather on Telegram → /newbot → get bot_token
    2. Message your bot → get chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
    3. Put bot_token and chat_id in config/credentials.yaml
    """

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self._enabled = bool(bot_token and chat_id)

        if self._enabled:
            self._init_bot()

    def _init_bot(self):
        """Initialize Telegram bot."""
        try:
            import requests
            self._requests = requests
            # Test connection
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                bot_info = response.json().get("result", {})
                logger.info(f"Telegram bot connected: @{bot_info.get('username', 'unknown')}")
            else:
                logger.warning(f"Telegram bot connection failed: {response.status_code}")
                self._enabled = False
        except ImportError:
            logger.warning("requests not available for Telegram bot")
            self._enabled = False
        except Exception as e:
            logger.warning(f"Telegram bot init failed: {e}")
            self._enabled = False

    def send_message(self, text: str, parse_mode: str = "HTML"):
        """Send a text message via Telegram."""
        if not self._enabled:
            logger.debug(f"[TG not active] Would send: {text[:50]}...")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = self._requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    # -----------------------------------------------------------------------
    # Pre-formatted alert methods
    # -----------------------------------------------------------------------

    def alert_new_signal(self, signal: Dict):
        """Send a new trading signal alert."""
        action = signal.get("action", "HOLD")
        symbol = signal.get("symbol", "")
        confidence = signal.get("confidence", 0)
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        target = signal.get("target", 0)
        rr = signal.get("risk_reward", 0)
        regime = signal.get("regime", "")
        reasoning = signal.get("reasoning", "")

        if action == "HOLD":
            return

        emoji = "🟢" if "CE" in action else "🔴" if "PE" in action else "🟡"

        text = (
            f"{emoji} <b>NEW SIGNAL</b>\n\n"
            f"<b>{action}</b> {symbol}\n"
            f"Confidence: {confidence:.0%} | R:R = 1:{rr:.1f}\n\n"
            f"Entry: <b>{entry:.2f}</b>\n"
            f"Stop Loss: <b>{sl:.2f}</b>\n"
            f"Target: <b>{target:.2f}</b>\n"
            f"Regime: {regime}\n\n"
            f"<i>{reasoning[:200]}</i>"
        )
        self.send_message(text)

    def alert_order_placed(self, order_info: Dict):
        """Alert when an order is placed."""
        text = (
            f"📋 <b>ORDER PLACED</b>\n\n"
            f"{order_info.get('side', '')} {order_info.get('quantity', 0)} "
            f"{order_info.get('symbol', '')}\n"
            f"Type: {order_info.get('order_type', '')}\n"
            f"ID: {order_info.get('order_id', '')}"
        )
        self.send_message(text)

    def alert_order_filled(self, order_info: Dict):
        """Alert when an order is filled."""
        text = (
            f"✅ <b>ORDER FILLED</b>\n\n"
            f"{order_info.get('side', '')} {order_info.get('quantity', 0)} "
            f"{order_info.get('symbol', '')}\n"
            f"Price: Rs {order_info.get('price', 0):.2f}\n"
            f"ID: {order_info.get('order_id', '')}"
        )
        self.send_message(text)

    def alert_stop_loss_hit(self, trade_info: Dict):
        """Alert when a stop loss is hit."""
        pnl = trade_info.get("pnl", 0)
        text = (
            f"🛑 <b>STOP LOSS HIT</b>\n\n"
            f"{trade_info.get('symbol', '')}\n"
            f"P&L: Rs {pnl:+,.0f}\n"
            f"Exit Price: {trade_info.get('exit_price', 0):.2f}"
        )
        self.send_message(text)

    def alert_target_hit(self, trade_info: Dict):
        """Alert when target is achieved."""
        pnl = trade_info.get("pnl", 0)
        text = (
            f"🎯 <b>TARGET HIT</b>\n\n"
            f"{trade_info.get('symbol', '')}\n"
            f"P&L: Rs {pnl:+,.0f} ✨\n"
            f"Exit Price: {trade_info.get('exit_price', 0):.2f}"
        )
        self.send_message(text)

    def alert_risk_breach(self, risk_info: Dict):
        """Alert when a risk limit is breached."""
        text = (
            f"⚠️ <b>RISK ALERT</b>\n\n"
            f"Violation: {risk_info.get('violation', '')}\n"
            f"Details: {risk_info.get('details', '')}\n"
            f"Action: {risk_info.get('action', 'Review immediately')}"
        )
        self.send_message(text)

    def alert_daily_summary(self, summary: Dict):
        """Send end-of-day summary."""
        pnl = summary.get("daily_pnl", 0)
        trades = summary.get("total_trades", 0)
        wins = summary.get("winning_trades", 0)
        equity = summary.get("equity", 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        text = (
            f"{pnl_emoji} <b>DAILY SUMMARY</b>\n"
            f"{datetime.now().strftime('%d %b %Y')}\n\n"
            f"P&L: Rs {pnl:+,.0f}\n"
            f"Trades: {trades} (Won: {wins})\n"
            f"Win Rate: {wins/trades*100:.0f}%\n" if trades > 0 else ""
            f"Portfolio: Rs {equity:,.0f}\n"
        )
        self.send_message(text)

    def alert_system_start(self):
        """Alert when system starts."""
        text = (
            f"🚀 <b>PROMETHEUS STARTED</b>\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"System is online and monitoring."
        )
        self.send_message(text)

    def alert_system_error(self, error: str):
        """Alert on critical system error."""
        text = (
            f"🔥 <b>SYSTEM ERROR</b>\n\n"
            f"{error[:300]}"
        )
        self.send_message(text)
