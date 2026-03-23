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
        instrument = signal.get("instrument", "")
        strike = signal.get("strike", 0)
        option_type = signal.get("option_type", "")
        expiry = signal.get("expiry", "")
        confidence = signal.get("confidence", 0)
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        target = signal.get("target", 0)
        rr = signal.get("risk_reward", 0)
        regime = signal.get("regime", "")
        reasoning = signal.get("reasoning", "")

        if action == "HOLD":
            return

        emoji = "\U0001f7e2" if "CE" in action else "\U0001f534"
        direction = "BULLISH" if "CE" in action else "BEARISH"

        quality, wr = REGIME_QUALITY.get(regime, ("???", ""))
        caution = ""
        if quality == "WEAK":
            caution = "\n\u26a0\ufe0f Low-confidence regime (26% WR)"
        elif quality == "LOW":
            caution = "\n\u26a0\ufe0f Volatile regime — lower conviction"

        contract_line = ""
        if instrument:
            contract_line = f"Contract      <code>{instrument}</code>\n"
        elif strike and option_type:
            exp = f" {expiry}" if expiry else ""
            contract_line = f"Contract      <code>{symbol}{exp} {int(float(strike))}{option_type}</code>\n"

        text = (
            f"{emoji} <b>SIGNAL  \u2014  {direction}</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{symbol}</b>  \u2502  {action}\n"
            f"Confidence  <code>{confidence:.0%}</code>  \u2502  R:R  <code>1:{rr:.1f}</code>\n\n"
            f"{contract_line}"
            f"\u25B6  Entry     <code>Rs {entry:,.2f}</code>\n"
            f"\U0001f6d1  Stop       <code>Rs {sl:,.2f}</code>\n"
            f"\U0001f3af  Target    <code>Rs {target:,.2f}</code>\n\n"
            f"\U0001f30d  {regime.upper()} ({quality} \u2014 {wr})"
            f"{caution}\n"
        )
        if reasoning:
            text += f"\n<i>{reasoning[:200]}</i>"

        self.send_message(text)

    def alert_scanner_summary(self, scan_results: List[Dict]):
        """Send multi-index scanner results as a formatted summary."""
        if not scan_results:
            self.send_message(
                "\U0001f50d <b>SCAN COMPLETE</b>\n"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
                "No actionable signals found."
            )
            return

        results = sorted(scan_results, key=lambda x: x.get("adj_confidence", 0), reverse=True)

        # Separate swing and intraday
        swing = [r for r in results if r.get("timeframe") != "intraday"]
        intraday = [r for r in results if r.get("timeframe") == "intraday"]

        lines = [
            f"\U0001f50d <b>MARKET SCAN</b>",
            f"<code>{datetime.now().strftime('%d %b %Y  %H:%M')}</code>",
            "",
        ]

        def _format_row(r, idx):
            action = r.get("action", "HOLD")
            symbol = r.get("symbol", "")
            regime = r.get("regime", "unknown")
            adj_conf = r.get("adj_confidence", 0)
            sig_count = r.get("signal_count", 0)
            quality, _ = REGIME_QUALITY.get(regime, ("???", ""))

            if "CE" in action:
                d_emoji = "\U0001f7e2"
            elif "PE" in action:
                d_emoji = "\U0001f534"
            else:
                d_emoji = "\u26aa"

            if quality == "HIGH":
                q_tag = "\u2705"
            elif quality == "MED":
                q_tag = "\U0001f7e1"
            elif quality == "LOW":
                q_tag = "\u26a0\ufe0f"
            else:
                q_tag = "\u274c"

            sig_only = "  <i>(signal only)</i>" if not r.get("executable", True) else ""

            return (
                f"{d_emoji} <b>{symbol}</b>{sig_only}\n"
                f"    {action}  \u2502  <code>{adj_conf:>3.0%}</code>  \u2502  {sig_count}/10 signals\n"
                f"    {q_tag} {regime.upper()} ({quality})"
            )

        if intraday:
            lines.append("\U0001f552 <b>INTRADAY</b>")
            lines.append("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
            for i, r in enumerate(intraday, 1):
                lines.append(_format_row(r, i))
            lines.append("")

        if swing:
            lines.append("\U0001f4c5 <b>SWING</b>")
            lines.append("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
            for i, r in enumerate(swing, 1):
                lines.append(_format_row(r, i))
            lines.append("")

        actionable = [r for r in results if r["action"] != "HOLD" and r.get("adj_confidence", 0) >= 0.50]
        lines.append("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        if actionable:
            lines.append(f"\U0001f525 <b>{len(actionable)} actionable signal(s)</b> above 50%")
        else:
            lines.append("\u23f8 No signals above 50% threshold")

        self.send_message("\n".join(lines))

        for r in actionable[:3]:
            self.alert_new_signal({
                "action": r["action"],
                "symbol": r["symbol"],
                "instrument": r.get("instrument", ""),
                "strike": r.get("strike", 0),
                "option_type": r.get("option_type", ""),
                "expiry": r.get("expiry", ""),
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
            f"\U0001f4cb <b>ORDER PLACED</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{order_info.get('side', '')}  {order_info.get('quantity', 0)}x</b>  "
            f"{order_info.get('symbol', '')}\n"
            f"Type: {order_info.get('order_type', '')}  \u2502  ID: <code>{order_info.get('order_id', '')}</code>"
        )
        self.send_message(text)

    def alert_order_filled(self, order_info: Dict):
        """Alert when an order is filled."""
        text = (
            f"\u2705 <b>ORDER FILLED</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{order_info.get('side', '')}  {order_info.get('quantity', 0)}x</b>  "
            f"{order_info.get('symbol', '')}\n"
            f"Price: <code>Rs {order_info.get('price', 0):,.2f}</code>  \u2502  "
            f"ID: <code>{order_info.get('order_id', '')}</code>"
        )
        self.send_message(text)

    def alert_stop_loss_hit(self, trade_info: Dict):
        """Alert when a stop loss is hit."""
        pnl = trade_info.get("pnl", 0)
        text = (
            f"\U0001f6d1 <b>STOP LOSS HIT</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{trade_info.get('symbol', '')}</b>\n"
            f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
            f"P&L: <code>Rs {pnl:+,.0f}</code>"
        )
        self.send_message(text)

    def alert_target_hit(self, trade_info: Dict):
        """Alert when target is achieved."""
        pnl = trade_info.get("pnl", 0)
        text = (
            f"\U0001f3af <b>TARGET HIT</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{trade_info.get('symbol', '')}</b>\n"
            f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
            f"P&L: <code>Rs {pnl:+,.0f}</code>"
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

        text = (
            f"{pnl_emoji} <b>TRADE CLOSED  \u2014  {result}</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{side}  {qty}x  {symbol}</b>\n"
            f"Exit: <code>Rs {price:,.2f}</code>\n\n"
            f"Gross P&L:  <code>Rs {gross_pnl:+,.2f}</code>\n"
            f"<b>Net P&L:    <code>Rs {net_pnl:+,.2f}</code></b>\n"
        )

        if costs:
            total_cost = costs.get('total', 0)
            text += (
                f"\n<i>Costs: Rs {total_cost:,.2f}</i>\n"
                f"<i>(Brokerage {costs.get('brokerage', 0):.1f} + "
                f"STT {costs.get('stt', 0):.1f} + "
                f"GST {costs.get('gst', 0):.1f} + others)</i>\n"
            )

        text += (
            f"\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\U0001f4b0 Portfolio: <b><code>Rs {equity:,.0f}</code></b>"
        )
        self.send_message(text)

    def alert_risk_breach(self, risk_info: Dict):
        """Alert when a risk limit is breached."""
        text = (
            f"\u26a0\ufe0f <b>RISK ALERT</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<b>{risk_info.get('violation', '')}</b>\n"
            f"{risk_info.get('details', '')}\n\n"
            f"Action: <i>{risk_info.get('action', 'Review immediately')}</i>"
        )
        self.send_message(text)

    def alert_daily_summary(self, summary: Dict):
        """Send end-of-day summary."""
        pnl = summary.get("daily_pnl", 0)
        trades = summary.get("total_trades", 0)
        wins = summary.get("winning_trades", 0)
        equity = summary.get("equity", 0)
        pnl_emoji = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"
        total_costs = summary.get("total_costs", 0)
        gross_pnl = summary.get("gross_pnl", pnl + total_costs)
        guardrail_audit = summary.get("intraday_guardrail_audit", "")

        wr_line = f"Win Rate:    <code>{wins/trades*100:.0f}%</code>\n" if trades > 0 else ""

        text = (
            f"{pnl_emoji} <b>DAILY SUMMARY</b>\n"
            f"<code>{datetime.now().strftime('%d %b %Y')}</code>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"Gross P&L:   <code>Rs {gross_pnl:+,.0f}</code>\n"
        )
        if total_costs > 0:
            text += f"Costs:       <code>Rs {total_costs:,.0f}</code>\n"
        text += (
            f"<b>Net P&L:     <code>Rs {pnl:+,.0f}</code></b>\n\n"
            f"Trades:      <code>{trades}</code>  (Won: {wins})\n"
            f"{wr_line}"
        )
        if guardrail_audit:
            text += f"Intraday Guardrail: <code>{guardrail_audit}</code>\n"
        text += (
            f"\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\U0001f4b0 Portfolio: <b><code>Rs {equity:,.0f}</code></b>"
        )
        self.send_message(text)

    def alert_system_start(self):
        """Alert when system starts."""
        text = (
            f"\U0001f680 <b>PROMETHEUS  ONLINE</b>\n"
            f"<code>{datetime.now().strftime('%d %b %Y  %H:%M')}</code>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"System is monitoring the markets.\n\n"
            f"/scan  \u2502  /status  \u2502  /pnl\n"
            f"/positions  \u2502  /regime  \u2502  /help"
        )
        self.send_message(text)

    def alert_system_error(self, error: str):
        """Alert on critical system error (non-blocking)."""
        text = (
            f"\U0001f525 <b>SYSTEM ERROR</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            f"<code>{error[:300]}</code>"
        )
        self.send_message_async(text)
