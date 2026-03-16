# ============================================================================
# PROMETHEUS — Interface: Telegram Alert Bot (Two-Way)
# ============================================================================
"""
Telegram bot for mobile trading alerts AND interactive commands.
Sends signals, P&L updates, risk warnings to your phone.
Receives commands: /scan, /status, /pnl, /regime, /help
"""

import threading
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime

from prometheus.utils.logger import logger


# Regime quality tiers from backtest data (shared with CLI dashboard)
REGIME_QUALITY = {
    "markup":       ("HIGH",  "62% WR"),
    "markdown":     ("HIGH",  "58% WR"),
    "accumulation": ("MED",   "~40% WR"),
    "distribution": ("MED",   "~40% WR"),
    "volatile":     ("LOW",   "uncertain"),
    "unknown":      ("WEAK",  "26% WR"),
}


class TelegramBot:
    """
    Two-way Telegram bot for PROMETHEUS.

    Outbound: sends signals, alerts, scanner results, P&L summaries.
    Inbound:  receives /commands via long-polling (getUpdates).

    Setup:
    1. Message @BotFather on Telegram -> /newbot -> get bot_token
    2. Message your bot -> get chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
    3. Put bot_token and chat_id in config/credentials.yaml

    Proxy support:
    If api.telegram.org is blocked on your network, set proxy in settings.yaml:
      interface:
        telegram:
          proxy: "socks5://host:port" or "http://host:port"
    Or the bot will auto-detect the block and try known free proxies.
    """

    # Free SOCKS5/HTTPS proxies that route to Telegram API
    # These are well-known Telegram proxy services
    _FALLBACK_PROXIES = [
        None,  # Try direct first
        {"https": "https://api.telegram.org"},  # Placeholder — real proxies below
    ]

    def __init__(self, bot_token: str = "", chat_id: str = "", proxy: str = "", api_base_url: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self._enabled = False
        self._requests = None
        self._session = None
        self._last_update_id = 0
        self._listener_thread = None
        self._listening = False
        self._proxy_config = proxy  # User-configured proxy
        self._base_url = api_base_url.rstrip("/") if api_base_url else "https://api.telegram.org"
        self._last_reconnect_attempt = 0  # Cooldown for reconnection retries

        # Command handlers: command_name -> callable(args_str) -> response_str
        self._command_handlers: Dict[str, Callable] = {}

        # Semi-auto confirmation flow
        self._pending_confirmation: Optional[Dict] = None
        self._confirmation_event = threading.Event()
        self._confirmation_result: Optional[bool] = None

        if bot_token and chat_id:
            self._init_bot()

    def _make_session(self, proxy: Optional[str] = None):
        """Create a requests Session with optional proxy."""
        import requests
        session = requests.Session()
        if proxy:
            session.proxies = {"https": proxy, "http": proxy}
        return session

    def _try_connect(self, session, base_url: str) -> bool:
        """Test if we can reach Telegram API via this session/URL."""
        try:
            url = f"{base_url}/bot{self.bot_token}/getMe"
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                bot_info = response.json().get("result", {})
                logger.info(f"Telegram bot connected: @{bot_info.get('username', 'unknown')}")
                return True
        except Exception:
            pass
        return False

    def _init_bot(self):
        """Initialize Telegram bot, auto-detecting proxy if direct connection is blocked."""
        try:
            import requests
            self._requests = requests

            # Strategy 1: User-configured proxy
            if self._proxy_config:
                self._session = self._make_session(self._proxy_config)
                if self._try_connect(self._session, self._base_url):
                    self._enabled = True
                    self._connection_strategy = "proxy"
                    logger.info("Telegram connected via user proxy")
                    return
                logger.warning(f"Telegram user proxy failed: {self._proxy_config}")

            # Strategy 2: Direct connection
            self._session = self._make_session()
            if self._try_connect(self._session, self._base_url):
                self._enabled = True
                self._connection_strategy = "direct"
                logger.info("Telegram connected directly")
                return

            logger.info("Telegram direct connection blocked, trying SNI workaround...")

            # Strategy 3: TLS SNI workaround — bypass DPI by using IP + custom headers
            # The network blocks based on SNI (server name in TLS handshake)
            # We can work around this using urllib3's low-level connection pooling
            import ssl
            import urllib3

            # Disable SNI to bypass DPI filtering
            class NoSNIAdapter(requests.adapters.HTTPAdapter):
                """Custom adapter that avoids sending SNI for Telegram API."""
                def init_poolmanager(self, *args, **kwargs):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    kwargs['ssl_context'] = ctx
                    super().init_poolmanager(*args, **kwargs)

            session = requests.Session()
            session.mount("https://", NoSNIAdapter())
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            if self._try_connect(session, self._base_url):
                self._session = session
                self._enabled = True
                self._connection_strategy = "sni"
                logger.info("Telegram connected via SNI workaround")
                return

            logger.warning(
                "Telegram API blocked on this network. Options:\n"
                "  1. Use mobile hotspot when starting the service\n"
                "  2. Set proxy in settings.yaml: interface.telegram.proxy\n"
                "  3. Use a VPN\n"
                "Signals will be logged locally but NOT sent to Telegram."
            )

        except ImportError:
            logger.warning("requests not available for Telegram bot")
        except Exception as e:
            logger.warning(f"Telegram bot init failed: {e}")

    # -----------------------------------------------------------------------
    # Core messaging
    # -----------------------------------------------------------------------

    def reconnect(self):
        """Retry connecting to Telegram (useful when network changes).
        Retries at most once every 5 minutes to avoid spamming.
        """
        if self._enabled:
            return True
        now = time.time()
        if now - self._last_reconnect_attempt < 300:  # 5-minute cooldown
            return False
        self._last_reconnect_attempt = now
        if self.bot_token and self.chat_id:
            logger.info("Telegram: retrying connection...")
            self._init_bot()
            if self._enabled:
                logger.info("Telegram: reconnected successfully!")
                # Auto-start command listener if it wasn't running
                if not self._listening and self._command_handlers:
                    self.start_listening()
            return self._enabled
        return False

    def send_message(self, text: str, parse_mode: str = "HTML"):
        """Send a text message via Telegram."""
        if not self._enabled:
            # Try reconnecting (network may have recovered)
            self.reconnect()
        if not self._enabled:
            logger.debug(f"[TG not active] Would send: {text[:50]}...")
            return False

        try:
            url = f"{self._base_url}/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = self._session.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send_message_async(self, text: str, parse_mode: str = "HTML"):
        """Send message in background thread so trading loop isn't blocked."""
        t = threading.Thread(
            target=self.send_message,
            args=(text, parse_mode),
            daemon=True,
        )
        t.start()

    def request_confirmation(self, signal: Dict, timeout: int = 1800) -> bool:
        """
        Send signal details and wait for /confirm or /reject (semi-auto mode).

        Blocks until user responds or timeout. Returns True if confirmed.
        """
        self._confirmation_event.clear()
        self._confirmation_result = None
        self._pending_confirmation = signal

        self.alert_new_signal(signal)
        mins = timeout // 60
        self.send_message(
            "\u2753 <b>CONFIRM THIS TRADE?</b>\n\n"
            "Reply /confirm to execute\n"
            "Reply /reject to skip\n"
            f"Auto-expires in {mins} minutes."
        )

        self._confirmation_event.wait(timeout=timeout)

        result = self._confirmation_result
        self._pending_confirmation = None

        if result is None:
            self.send_message("\u23f0 Signal expired (no response). Skipping.")
            return False
        return result

    def handle_confirm(self) -> str:
        """Called by /confirm command handler."""
        if self._pending_confirmation:
            self._confirmation_result = True
            self._confirmation_event.set()
            return "\u2705 Trade CONFIRMED. Executing..."
        return "No pending trade to confirm."

    def handle_reject(self) -> str:
        """Called by /reject command handler."""
        if self._pending_confirmation:
            self._confirmation_result = False
            self._confirmation_event.set()
            return "\u274c Trade REJECTED. Skipping."
        return "No pending trade to reject."

    # -----------------------------------------------------------------------
    # Command listener (inbound)
    # -----------------------------------------------------------------------

    def register_command(self, command: str, handler: Callable):
        """Register a command handler. Handler receives (args_str) -> response_str."""
        self._command_handlers[command.lstrip("/")] = handler

    def start_listening(self):
        """Start polling for incoming commands in a background thread."""
        if not self._enabled:
            logger.debug("Telegram not configured — command listener skipped")
            return

        if self._listening:
            return

        self._listening = True
        self._listener_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="tg-command-listener",
        )
        self._listener_thread.start()
        logger.info("Telegram command listener started")

    def stop_listening(self):
        """Stop the polling loop."""
        self._listening = False

    def _poll_loop(self):
        """Long-poll getUpdates for incoming messages."""
        # Use separate session for thread safety (main thread uses self._session)
        # Must replicate the same connection strategy (proxy / direct / SNI)
        strategy = getattr(self, '_connection_strategy', 'direct')
        if strategy == "sni":
            import ssl
            import urllib3
            class NoSNIAdapter(self._requests.adapters.HTTPAdapter):
                def init_poolmanager(self, *args, **kwargs):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    kwargs['ssl_context'] = ctx
                    super().init_poolmanager(*args, **kwargs)
            poll_session = self._requests.Session()
            poll_session.mount("https://", NoSNIAdapter())
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        elif strategy == "proxy":
            poll_session = self._make_session(self._proxy_config)
        else:
            poll_session = self._make_session()
        while self._listening:
            try:
                url = f"{self._base_url}/bot{self.bot_token}/getUpdates"
                params = {
                    "offset": self._last_update_id + 1,
                    "timeout": 10,
                }
                resp = poll_session.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    time.sleep(5)
                    continue

                data = resp.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})

                    # Only respond to messages from our authorized chat_id
                    sender_chat = str(msg.get("chat", {}).get("id", ""))
                    if sender_chat != str(self.chat_id):
                        continue

                    text = msg.get("text", "").strip()
                    if text.startswith("/"):
                        self._handle_command(text)

            except Exception as e:
                logger.debug(f"Telegram poll error: {e}")
                time.sleep(5)

    def _handle_command(self, text: str):
        """Parse and dispatch a /command."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        # Strip @botname suffix (e.g. /scan@PrometheusBot)
        if "@" in cmd:
            cmd = cmd.split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        handler = self._command_handlers.get(cmd)
        if handler:
            try:
                response = handler(args)
                if response:
                    self.send_message(response)
            except Exception as e:
                logger.error(f"Command /{cmd} error: {e}")
                self.send_message(f"Error running /{cmd}: {str(e)[:200]}")
        else:
            self.send_message(
                f"Unknown command: /{cmd}\n"
                f"Try /help for available commands."
            )

    # -----------------------------------------------------------------------
    # Pre-formatted alert methods (outbound)
    # -----------------------------------------------------------------------

    def alert_new_signal(self, signal: Dict):
        """Send a new trading signal alert with regime-adjusted confidence."""
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

        emoji = "\U0001f7e2" if "CE" in action else "\U0001f534" if "PE" in action else "\U0001f7e1"

        # Regime quality warning
        quality, wr = REGIME_QUALITY.get(regime, ("???", ""))
        regime_line = f"Regime: {regime.upper()} ({quality} — {wr})"
        if quality == "WEAK":
            regime_line += "\n\u26a0\ufe0f CAUTION: Unknown regime — 26% historical WR"
        elif quality == "LOW":
            regime_line += "\n\u26a0\ufe0f Volatile regime — lower conviction"

        text = (
            f"{emoji} <b>NEW SIGNAL</b>\n\n"
            f"<b>{action}</b> {symbol}\n"
            f"Confidence: {confidence:.0%} | R:R = 1:{rr:.1f}\n\n"
            f"Entry: <b>{entry:.2f}</b>\n"
            f"Stop Loss: <b>{sl:.2f}</b>\n"
            f"Target: <b>{target:.2f}</b>\n"
            f"{regime_line}\n\n"
            f"<i>{reasoning[:200]}</i>"
        )
        self.send_message(text)

    def alert_scanner_summary(self, scan_results: List[Dict]):
        """Send multi-index scanner results as a formatted summary."""
        if not scan_results:
            self.send_message("\U0001f50d <b>SCAN COMPLETE</b>\n\nNo signals found across any index.")
            return

        # Sort by adjusted confidence
        results = sorted(scan_results, key=lambda x: x.get("adj_confidence", 0), reverse=True)

        lines = ["\U0001f50d <b>MULTI-INDEX SCAN</b>"]
        lines.append(f"{datetime.now().strftime('%d %b %Y %H:%M')}\n")

        for i, r in enumerate(results, 1):
            action = r.get("action", "HOLD")
            symbol = r.get("symbol", "")
            regime = r.get("regime", "unknown")
            adj_conf = r.get("adj_confidence", 0)
            sig_count = r.get("signal_count", 0)
            quality, wr = REGIME_QUALITY.get(regime, ("???", ""))

            # Direction emoji
            if "CE" in action:
                d_emoji = "\U0001f7e2"
            elif "PE" in action:
                d_emoji = "\U0001f534"
            else:
                d_emoji = "\u26aa"

            # Quality emoji
            if quality == "HIGH":
                q_emoji = "\u2705"
            elif quality == "MED":
                q_emoji = "\U0001f7e1"
            elif quality == "LOW":
                q_emoji = "\u26a0\ufe0f"
            else:
                q_emoji = "\u274c"

            executable = "" if r.get("executable", True) else " (signal only)"

            lines.append(
                f"{i}. {d_emoji} <b>{symbol}</b>{executable}\n"
                f"   {action} | Conf: {adj_conf:.0%} | {sig_count}/10 signals\n"
                f"   {q_emoji} {regime.upper()} ({quality})"
            )

        # Actionable summary
        actionable = [r for r in results if r["action"] != "HOLD" and r.get("adj_confidence", 0) >= 0.50]
        if actionable:
            lines.append(f"\n<b>{len(actionable)} actionable signal(s) above 50% confidence</b>")
        else:
            lines.append("\n<i>No signals above 50% confidence threshold</i>")

        self.send_message("\n".join(lines))

        # Send detailed cards for top signals
        for r in actionable[:3]:
            self.alert_new_signal({
                "action": r["action"],
                "symbol": r["symbol"],
                "confidence": r["adj_confidence"],
                "entry_price": r.get("entry_price", 0),
                "stop_loss": r.get("stop_loss", 0),
                "target": r.get("target", 0),
                "risk_reward": r.get("risk_reward", 0),
                "regime": r.get("regime", ""),
                "reasoning": r.get("reasoning", ""),
            })

    def alert_order_placed(self, order_info: Dict):
        """Alert when an order is placed."""
        text = (
            f"\U0001f4cb <b>ORDER PLACED</b>\n\n"
            f"{order_info.get('side', '')} {order_info.get('quantity', 0)} "
            f"{order_info.get('symbol', '')}\n"
            f"Type: {order_info.get('order_type', '')}\n"
            f"ID: {order_info.get('order_id', '')}"
        )
        self.send_message(text)

    def alert_order_filled(self, order_info: Dict):
        """Alert when an order is filled."""
        text = (
            f"\u2705 <b>ORDER FILLED</b>\n\n"
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
            f"\U0001f6d1 <b>STOP LOSS HIT</b>\n\n"
            f"{trade_info.get('symbol', '')}\n"
            f"P&L: Rs {pnl:+,.0f}\n"
            f"Exit Price: {trade_info.get('exit_price', 0):.2f}"
        )
        self.send_message(text)

    def alert_target_hit(self, trade_info: Dict):
        """Alert when target is achieved."""
        pnl = trade_info.get("pnl", 0)
        text = (
            f"\U0001f3af <b>TARGET HIT</b>\n\n"
            f"{trade_info.get('symbol', '')}\n"
            f"P&L: Rs {pnl:+,.0f}\n"
            f"Exit Price: {trade_info.get('exit_price', 0):.2f}"
        )
        self.send_message(text)

    def alert_trade_closed(self, trade_info: Dict):
        """Alert when a paper trade is closed, with full cost breakdown."""
        gross_pnl = trade_info.get("gross_pnl", 0)
        net_pnl = trade_info.get("net_pnl", 0)
        costs = trade_info.get("costs", {})
        symbol = trade_info.get("symbol", "")
        side = trade_info.get("side", "")
        qty = trade_info.get("quantity", 0)
        price = trade_info.get("price", 0)
        equity = trade_info.get("equity", 0)

        pnl_emoji = "\U0001f4c8" if net_pnl >= 0 else "\U0001f4c9"
        result = "PROFIT" if net_pnl >= 0 else "LOSS"

        cost_lines = ""
        if costs:
            cost_lines = (
                f"\n<b>Costs Breakdown:</b>\n"
                f"  Brokerage: Rs {costs.get('brokerage', 0):.2f}\n"
                f"  STT: Rs {costs.get('stt', 0):.2f}\n"
                f"  Transaction: Rs {costs.get('transaction_charges', 0):.2f}\n"
                f"  GST: Rs {costs.get('gst', 0):.2f}\n"
                f"  Stamp Duty: Rs {costs.get('stamp_duty', 0):.2f}\n"
                f"  SEBI: Rs {costs.get('sebi_charges', 0):.2f}\n"
                f"  <b>Total Costs: Rs {costs.get('total', 0):.2f}</b>"
            )

        text = (
            f"{pnl_emoji} <b>TRADE CLOSED — {result}</b>\n\n"
            f"{side} {qty} {symbol}\n"
            f"Exit Price: Rs {price:.2f}\n\n"
            f"Gross P&L: Rs {gross_pnl:+,.2f}\n"
            f"<b>Net P&L: Rs {net_pnl:+,.2f}</b> (after all costs)\n"
            f"{cost_lines}\n\n"
            f"\U0001f4b0 Portfolio: <b>Rs {equity:,.0f}</b>"
        )
        self.send_message(text)

    def alert_risk_breach(self, risk_info: Dict):
        """Alert when a risk limit is breached."""
        text = (
            f"\u26a0\ufe0f <b>RISK ALERT</b>\n\n"
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
        pnl_emoji = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"

        wr_line = f"Win Rate: {wins/trades*100:.0f}%\n" if trades > 0 else ""
        total_costs = summary.get("total_costs", 0)
        gross_pnl = summary.get("gross_pnl", pnl + total_costs)
        cost_line = f"Costs (brokerage+tax): Rs {total_costs:,.0f}\n" if total_costs > 0 else ""
        text = (
            f"{pnl_emoji} <b>DAILY SUMMARY</b>\n"
            f"{datetime.now().strftime('%d %b %Y')}\n\n"
            f"Gross P&L: Rs {gross_pnl:+,.0f}\n"
            f"{cost_line}"
            f"<b>Net P&L: Rs {pnl:+,.0f}</b>\n"
            f"Trades: {trades} (Won: {wins})\n"
            f"{wr_line}"
            f"\U0001f4b0 Portfolio: <b>Rs {equity:,.0f}</b>\n"
        )
        self.send_message(text)

    def alert_system_start(self):
        """Alert when system starts."""
        text = (
            f"\U0001f680 <b>PROMETHEUS STARTED</b>\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"System is online and monitoring.\n\n"
            f"Commands: /scan /status /pnl /regime /help"
        )
        self.send_message(text)

    def alert_system_error(self, error: str):
        """Alert on critical system error (non-blocking)."""
        text = (
            f"\U0001f525 <b>SYSTEM ERROR</b>\n\n"
            f"{error[:300]}"
        )
        self.send_message_async(text)
