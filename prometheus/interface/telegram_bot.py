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
import queue
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
        self._send_lock = threading.Lock()
        self._send_queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=200)
        self._send_worker_thread = None
        self._send_worker_running = False
        self._send_fail_streak = 0
        self._last_send_ok_ts = time.time()
        self._last_update_id = 0
        self._listener_thread = None
        self._listening = False
        self._poll_reset_requested = threading.Event()
        self._watchdog_thread = None
        self._watchdog_running = False
        self._watchdog_interval_sec = 60
        self._health_ping_interval_sec = 180
        self._last_health_ping_ts = 0.0
        self._last_command_rx_ts = 0.0
        self._last_poll_conflict_log_ts = 0.0
        self._poll_conflict_cooldown_sec = 20
        self._last_transport_recovery_ts = 0.0
        self._transport_recovery_cooldown_sec = 90
        self._proxy_config = proxy
        self._preferred_base_url = api_base_url.rstrip("/") if api_base_url else ""
        self._base_url = self._preferred_base_url or "https://api.telegram.org"
        self._last_reconnect_attempt = 0
        self._connection_strategy = "direct"

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

    def _ordered_base_candidates(self) -> List[str]:
        """Return API base URLs in preferred order for this environment."""
        official = "https://api.telegram.org"
        ordered: List[str] = []
        for base in [self._preferred_base_url, self._base_url, official]:
            if base and base not in ordered:
                ordered.append(base)
        return ordered or [official]

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

    def _build_session_for_strategy(self, strategy: str):
        """Create a requests session for the requested transport strategy."""
        if strategy == "sni":
            import ssl
            import urllib3

            class NoSNIAdapter(self._requests.adapters.HTTPAdapter):
                def init_poolmanager(self, *args, **kwargs):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    kwargs["ssl_context"] = ctx
                    super().init_poolmanager(*args, **kwargs)

            session = self._requests.Session()
            session.mount("https://", NoSNIAdapter())
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return session

        if strategy == "proxy" and self._proxy_config:
            return self._make_session(self._proxy_config)

        return self._make_session()

    def _advance_base_url(self) -> str:
        """Rotate to the next configured Telegram base URL."""
        candidates = self._ordered_base_candidates()
        if len(candidates) <= 1:
            return self._base_url
        try:
            idx = candidates.index(self._base_url)
        except ValueError:
            idx = 0
        self._base_url = candidates[(idx + 1) % len(candidates)]
        return self._base_url

    def _request_transport_recovery(self, reason: str):
        """Force a transport refresh before command replies are dropped."""
        if not self._requests:
            return
        now = time.time()
        if now - self._last_transport_recovery_ts < self._transport_recovery_cooldown_sec:
            logger.debug("Telegram transport recovery skipped (cooldown active)")
            return

        # 409 conflicts are usually a second poller; avoid rotating transports immediately.
        if "poll transport streak" in reason and (
            now - self._last_poll_conflict_log_ts < self._poll_conflict_cooldown_sec
        ):
            logger.debug("Telegram transport recovery skipped (recent polling conflict)")
            return

        old_base = self._base_url
        new_base = self._advance_base_url()
        try:
            self._session = self._build_session_for_strategy(self._connection_strategy)
            self._clear_webhook(self._session, new_base)
        except Exception as e:
            logger.warning(f"Telegram transport recovery session rebuild failed: {e}")
        self._poll_reset_requested.set()
        self._last_transport_recovery_ts = now
        logger.warning(
            f"Telegram transport recovery triggered ({reason}); "
            f"base {old_base} -> {new_base}"
        )

    def _probe_transport_health(self) -> bool:
        """Check if any configured Telegram endpoint is currently reachable."""
        if not self._requests:
            return False
        probe_session = self._build_session_for_strategy(self._connection_strategy)
        try:
            for base_url in self._ordered_base_candidates():
                if self._try_connect(probe_session, base_url):
                    self._base_url = base_url
                    return True
            return False
        finally:
            try:
                probe_session.close()
            except Exception:
                pass

    def _init_bot(self):
        """Initialize Telegram bot with proxy/direct/SNI and base URL fallback."""
        try:
            import requests
            import ssl
            import urllib3

            self._requests = requests

            base_candidates = self._ordered_base_candidates()

            # Strategy 1: user proxy
            if self._proxy_config:
                for base_url in base_candidates:
                    session = self._make_session(self._proxy_config)
                    if self._try_connect(session, base_url):
                        self._session = session
                        self._enabled = True
                        self._connection_strategy = "proxy"
                        self._base_url = base_url
                        self._clear_webhook(session, base_url)
                        self._ensure_send_worker()
                        logger.info(f"Telegram connected via user proxy (base: {base_url})")
                        return
                logger.warning(f"Telegram user proxy failed: {self._proxy_config}")

            # Strategy 2: direct
            for base_url in base_candidates:
                session = self._make_session()
                if self._try_connect(session, base_url):
                    self._session = session
                    self._enabled = True
                    self._connection_strategy = "direct"
                    self._base_url = base_url
                    self._clear_webhook(session, base_url)
                    self._ensure_send_worker()
                    logger.info(f"Telegram connected directly (base: {base_url})")
                    return

            logger.info("Telegram direct connection blocked, trying SNI workaround...")

            class NoSNIAdapter(requests.adapters.HTTPAdapter):
                """Custom adapter that avoids sending SNI for Telegram API."""

                def init_poolmanager(self, *args, **kwargs):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    kwargs["ssl_context"] = ctx
                    super().init_poolmanager(*args, **kwargs)

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            for base_url in base_candidates:
                session = requests.Session()
                session.mount("https://", NoSNIAdapter())
                if self._try_connect(session, base_url):
                    self._session = session
                    self._enabled = True
                    self._connection_strategy = "sni"
                    self._base_url = base_url
                    self._clear_webhook(session, base_url)
                    self._ensure_send_worker()
                    logger.info(f"Telegram connected via SNI workaround (base: {base_url})")
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

    def _clear_webhook(self, session, base_url: str):
        """Disable webhook mode to prevent getUpdates conflicts."""
        try:
            url = f"{base_url}/bot{self.bot_token}/deleteWebhook"
            payload = {"drop_pending_updates": False}
            resp = session.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram webhook cleared for long-poll mode")
            else:
                logger.debug(f"Telegram deleteWebhook returned {resp.status_code}")
        except Exception as e:
            logger.debug(f"Telegram deleteWebhook failed: {e}")

    def _ensure_send_worker(self):
        """Start a single background sender to avoid unbounded thread creation."""
        if self._send_worker_running and self._send_worker_thread and self._send_worker_thread.is_alive():
            return
        self._send_worker_running = True
        self._send_worker_thread = threading.Thread(
            target=self._send_worker_loop,
            daemon=True,
            name="tg-send-worker",
        )
        self._send_worker_thread.start()

    def _ensure_watchdog(self):
        """Start transport watchdog for periodic health checks."""
        if self._watchdog_running and self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="tg-transport-watchdog",
        )
        self._watchdog_thread.start()

    def _send_worker_loop(self):
        """Drain async message queue serially."""
        while self._send_worker_running:
            try:
                text, parse_mode = self._send_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self.send_message(text, parse_mode=parse_mode)
            finally:
                self._send_queue.task_done()

    def _watchdog_loop(self):
        """Periodic health checks and proactive transport recovery."""
        while self._watchdog_running:
            time.sleep(self._watchdog_interval_sec)
            if not self._enabled:
                continue
            now = time.time()
            if now - self._last_health_ping_ts < self._health_ping_interval_sec:
                continue
            self._last_health_ping_ts = now
            ok = self._probe_transport_health()
            if not ok:
                self._request_transport_recovery("watchdog health ping failed")

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

        with self._send_lock:
            try:
                base_candidates = self._ordered_base_candidates()

                for idx, base_url in enumerate(base_candidates):
                    url = f"{base_url}/bot{self.bot_token}/sendMessage"
                    payload = {
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                    }

                    try:
                        response = self._session.post(url, json=payload, timeout=10)
                    except Exception as e:
                        if idx + 1 < len(base_candidates):
                            logger.warning(
                                f"Telegram send transport error on {base_url}; "
                                f"retrying via {base_candidates[idx + 1]}"
                            )
                            continue
                        logger.error(f"Telegram send error: {e}")
                        return False

                    if response.status_code == 200:
                        self._send_fail_streak = 0
                        self._last_send_ok_ts = time.time()
                        return True

                    if response.status_code == 400 and parse_mode == "HTML":
                        # Fallback: retry as plain text when HTML formatting is rejected.
                        fallback_payload = {
                            "chat_id": self.chat_id,
                            "text": text,
                        }
                        retry = self._session.post(url, json=fallback_payload, timeout=10)
                        if retry.status_code == 200:
                            logger.warning("Telegram HTML payload rejected (400); sent as plain text fallback")
                            self._send_fail_streak = 0
                            self._last_send_ok_ts = time.time()
                            return True

                    if idx + 1 < len(base_candidates):
                        logger.warning(
                            f"Telegram send failed on {base_url} ({response.status_code}); "
                            f"retrying via {base_candidates[idx + 1]}"
                        )
                        continue

                    logger.warning(f"Telegram send failed: {response.status_code}")
                    self._send_fail_streak += 1
                    if self._send_fail_streak >= 3:
                        self._request_transport_recovery("send failure streak")
                    return False
            except Exception as e:
                logger.error(f"Telegram send error: {e}")
                self._send_fail_streak += 1
                if self._send_fail_streak >= 3:
                    self._request_transport_recovery("send exception streak")
                return False

    def _send_command_reply(self, text: str):
        """Send command response with forced recovery + one retry if needed."""
        if self.send_message(text):
            return
        self._request_transport_recovery("command reply failed")
        if not self.send_message(text):
            logger.error("Telegram command reply dropped after recovery retry")

    def _send_command_reply_async(self, text: str):
        """Send command reply without blocking the polling loop."""
        t = threading.Thread(
            target=self._send_command_reply,
            args=(text,),
            daemon=True,
            name="tg-command-reply",
        )
        t.start()

    def send_message_async(self, text: str, parse_mode: str = "HTML"):
        """Send message in background thread so trading loop isn't blocked."""
        self._ensure_send_worker()
        try:
            self._send_queue.put_nowait((text, parse_mode))
        except queue.Full:
            logger.warning("Telegram async send queue full; falling back to direct send")
            self.send_message(text, parse_mode=parse_mode)

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
            # Try reconnect once before giving up; network may have recovered.
            self.reconnect()
        if not self._enabled:
            logger.debug("Telegram not configured — command listener skipped")
            return

        if self._listening:
            return

        self._ensure_send_worker()
        self._ensure_watchdog()

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

    def _build_poll_session(self, strategy: str):
        """Create a polling session based on current connection strategy."""
        return self._build_session_for_strategy(strategy)

    def _poll_loop(self):
        """Long-poll getUpdates for incoming messages."""
        # Use separate session for thread safety (main thread uses self._session)
        # Must replicate the same connection strategy (proxy / direct / SNI)
        strategy = getattr(self, '_connection_strategy', 'direct')
        poll_session = self._build_poll_session(strategy)

        # Some custom relay endpoints support sendMessage/getMe but not getUpdates.
        # For inbound commands, fall back to official API when relay polling fails.
        poll_base_candidates = self._ordered_base_candidates()
        poll_base_idx = 0
        consecutive_poll_errors = 0
        poll_error_streak = 0
        last_base_rotate_ts = 0.0
        base_rotate_cooldown_sec = 30

        def _rotate_poll_base(reason: str):
            nonlocal poll_base_idx, last_base_rotate_ts
            if len(poll_base_candidates) <= 1:
                return
            now = time.time()
            if now - last_base_rotate_ts < base_rotate_cooldown_sec:
                logger.debug("Telegram polling base rotation skipped (cooldown active)")
                return
            old = poll_base_candidates[poll_base_idx]
            poll_base_idx = (poll_base_idx + 1) % len(poll_base_candidates)
            new = poll_base_candidates[poll_base_idx]
            last_base_rotate_ts = now
            logger.warning(f"Telegram polling {reason} on {old}; switching getUpdates to {new}")

        while self._listening:
            try:
                if self._poll_reset_requested.is_set():
                    logger.info("Telegram polling session reset requested")
                    poll_session.close()
                    poll_session = self._build_poll_session(strategy)
                    self._poll_reset_requested.clear()

                poll_base = poll_base_candidates[poll_base_idx]
                url = f"{poll_base}/bot{self.bot_token}/getUpdates"
                params = {
                    "offset": self._last_update_id + 1,
                    "timeout": 8,
                }
                resp = poll_session.get(url, params=params, timeout=12)
                if resp.status_code != 200:
                    if resp.status_code == 409:
                        # Another bot instance is polling getUpdates.
                        # Treat as soft conflict: wait and retry without transport churn.
                        now = time.time()
                        if now - self._last_poll_conflict_log_ts >= self._poll_conflict_cooldown_sec:
                            logger.warning(
                                "Telegram polling conflict (409): another getUpdates session is active; "
                                "waiting before retry"
                            )
                            self._last_poll_conflict_log_ts = now
                        consecutive_poll_errors = 0
                        poll_error_streak = 0
                        time.sleep(8)
                        continue

                    consecutive_poll_errors += 1
                    poll_error_streak += 1
                    if consecutive_poll_errors >= 2:
                        _rotate_poll_base(f"failed ({resp.status_code})")
                        consecutive_poll_errors = 0
                    if poll_error_streak >= 6:
                        logger.warning("Telegram polling still failing; resetting poll session")
                        poll_session.close()
                        poll_session = self._build_poll_session(strategy)
                        poll_error_streak = 0
                    time.sleep(5)
                    continue

                consecutive_poll_errors = 0
                poll_error_streak = 0

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
                        self._last_command_rx_ts = time.time()
                        logger.info(f"Telegram command received: {text}")
                        self._handle_command(text)

            except Exception as e:
                logger.debug(f"Telegram poll error: {e}")
                consecutive_poll_errors += 1
                poll_error_streak += 1

                # Relay endpoints can pass getMe/sendMessage but fail getUpdates.
                # If repeated transport errors occur, rotate to next base candidate.
                if consecutive_poll_errors >= 2:
                    _rotate_poll_base("transport error")
                    consecutive_poll_errors = 0
                if poll_error_streak >= 6:
                    logger.warning("Telegram polling transport remains unstable; reconnecting listener session")
                    poll_session.close()
                    poll_session = self._build_poll_session(strategy)
                    poll_error_streak = 0
                elif poll_error_streak >= 3:
                    self._request_transport_recovery("poll transport streak")
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
                    self._send_command_reply_async(response)
            except Exception as e:
                logger.error(f"Command /{cmd} error: {e}")
                self._send_command_reply_async(f"Error running /{cmd}: {str(e)[:200]}")
        else:
            self._send_command_reply_async(
                f"Unknown command: /{cmd}\n"
                f"Try /help for available commands."
            )

    # -----------------------------------------------------------------------
    # Pre-formatted alert methods (outbound)
    # -----------------------------------------------------------------------

    def alert_new_signal(self, signal: Dict):
        """Send a compact mobile-friendly trading signal alert."""
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

        ranked = signal.get("eligible_strikes", [])
        if ranked:
            account_label = signal.get("account_label", "account")
            account_capital = float(signal.get("account_capital", 0) or 0)
            direction = "BULLISH" if "CE" in action else "BEARISH"
            lines = [
                f"\U0001f3af <b>{account_label} SIGNAL</b>",
                f"{symbol} | {direction}",
                f"Capital: <code>Rs {account_capital:,.0f}</code>",
                f"Confidence: <code>{confidence:.0%}</code>",
                "",
                "<b>Eligible (Best -> Worst)</b>",
            ]

            for i, c in enumerate(ranked, 1):
                lines.extend([
                    f"{i}) <code>{c.get('instrument', '')}</code> [{c.get('strike_tier', '')}]",
                    f"   Prem Rs {c.get('entry_price', 0):,.2f} | Cost Rs {c.get('lot_cost', 0):,.0f} | Lot {int(c.get('lot_size', 0))}",
                    f"   MinCap Rs {c.get('min_capital_required', 0):,.0f} | Risk1L Rs {c.get('risk_amount_1lot', 0):,.0f}",
                    f"   E {c.get('entry_price', 0):,.2f} | SL {c.get('stop_loss', 0):,.2f} | T {c.get('target', 0):,.2f} | RR 1:{c.get('risk_reward', 0):.2f}",
                    f"   Rec lots: {int(c.get('recommended_lots', 1))}",
                ])

            self.send_message("\n".join(lines))
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
            contract_line = f"Contract: <code>{instrument}</code>\n"
        elif strike and option_type:
            exp = f" {expiry}" if expiry else ""
            contract_line = f"Contract: <code>{symbol}{exp} {int(float(strike))}{option_type}</code>\n"

        lot_size = int(signal.get("lot_size", 0) or 0)
        lot_cost = float(signal.get("lot_cost", 0) or 0)
        min_cap = float(signal.get("min_capital_required", 0) or 0)
        risk_1lot = float(signal.get("risk_amount_1lot", 0) or 0)
        cost_line = ""
        if lot_size > 0 and lot_cost > 0:
            cost_line = f"Lot: <code>{lot_size}</code> | Cost: <code>Rs {lot_cost:,.0f}</code>\n"
            if min_cap > 0:
                cost_line += f"MinCap: <code>Rs {min_cap:,.0f}</code>\n"
            if risk_1lot > 0:
                cost_line += f"Risk(1 lot): <code>Rs {risk_1lot:,.0f}</code>\n"

        text = (
            f"{emoji} <b>NEW SIGNAL</b>\n"
            f"{symbol} | {action} | {direction}\n"
            f"Confidence: <code>{confidence:.0%}</code> | R:R: <code>1:{rr:.1f}</code>\n"
            f"{contract_line}"
            f"{cost_line}"
            f"Entry <code>Rs {entry:,.2f}</code> | SL <code>Rs {sl:,.2f}</code> | Target <code>Rs {target:,.2f}</code>\n"
            f"Regime: {regime.upper()} ({quality} — {wr})"
            f"{caution}\n"
        )
        if reasoning:
            text += f"\n<i>{reasoning[:200]}</i>"

        self.send_message(text)

    def alert_scanner_summary(self, scan_results: List[Dict]):
        """Send multi-index scanner results in a compact, mobile-friendly format."""
        if not scan_results:
            self.send_message(
                "\U0001f50d <b>SCAN COMPLETE</b>\n"
                "No actionable signals found."
            )
            return

        results = sorted(scan_results, key=lambda x: x.get("adj_confidence", 0), reverse=True)

        swing = [r for r in results if r.get("timeframe") != "intraday"]
        intraday = [r for r in results if r.get("timeframe") == "intraday"]
        swing_actionable = [r for r in swing if r.get("action") != "HOLD"]
        intraday_actionable = [r for r in intraday if r.get("action") != "HOLD"]
        swing_top = swing_actionable[:3]
        intraday_top = intraday_actionable[:3]

        lines = [
            "\U0001f50d <b>MARKET SCAN</b>",
            f"<code>{datetime.now().strftime('%d %b %Y  %H:%M')}</code>",
            "<i>Showing top-ranked actionable signals only.</i>",
            "",
        ]

        def _format_row(r):
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
                f"   {action} | <code>{adj_conf:>3.0%}</code> | {sig_count}/10\n"
                f"   {q_tag} {regime.upper()} ({quality})"
            )

        if intraday_top:
            lines.append("\U0001f552 <b>INTRADAY</b> <i>(configured instruments)</i>")
            for r in intraday_top:
                lines.append(_format_row(r))
            lines.append("")
        elif intraday:
            lines.append("\U0001f552 <b>INTRADAY</b> <i>(configured instruments)</i>")
            lines.append("\u23f8 No actionable intraday signals in the top-ranked set")
            lines.append("")

        if swing_top:
            lines.append("\U0001f4c5 <b>SWING</b> <i>(indices + stocks)</i>")
            for r in swing_top:
                lines.append(_format_row(r))
            lines.append("")
        elif swing:
            lines.append("\U0001f4c5 <b>SWING</b> <i>(indices + stocks)</i>")
            lines.append("\u23f8 No actionable swing signals in the top-ranked set")
            lines.append("")

        actionable = [r for r in results if r["action"] != "HOLD" and r.get("adj_confidence", 0) >= 0.50]
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
            "\U0001f4cb <b>ORDER PLACED</b>\n"
            f"{order_info.get('symbol', '')}\n"
            f"{order_info.get('side', '')} {order_info.get('quantity', 0)}x | {order_info.get('order_type', '')}\n"
            f"ID: <code>{order_info.get('order_id', '')}</code>"
        )
        self.send_message(text)

    def alert_order_filled(self, order_info: Dict):
        """Alert when an order is filled."""
        text = (
            "\u2705 <b>ORDER FILLED</b>\n"
            f"{order_info.get('symbol', '')}\n"
            f"{order_info.get('side', '')} {order_info.get('quantity', 0)}x\n"
            f"Price: <code>Rs {order_info.get('price', 0):,.2f}</code>\n"
            f"ID: <code>{order_info.get('order_id', '')}</code>"
        )
        self.send_message(text)

    def alert_stop_loss_hit(self, trade_info: Dict):
        """Alert when a stop loss is hit."""
        pnl = trade_info.get("pnl", 0)
        text = (
            "\U0001f6d1 <b>STOP LOSS HIT</b>\n"
            f"{trade_info.get('symbol', '')}\n"
            f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
            f"P&L: <code>Rs {pnl:+,.0f}</code>"
        )
        self.send_message(text)

    def alert_target_hit(self, trade_info: Dict):
        """Alert when target is achieved."""
        pnl = trade_info.get("pnl", 0)
        text = (
            "\U0001f3af <b>TARGET HIT</b>\n"
            f"{trade_info.get('symbol', '')}\n"
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
            f"{pnl_emoji} <b>TRADE CLOSED ({result})</b>\n"
            f"{side} {qty}x {symbol}\n"
            f"Exit: <code>Rs {price:,.2f}</code>\n\n"
            f"Gross P&L: <code>Rs {gross_pnl:+,.2f}</code>\n"
            f"<b>Net P&L: <code>Rs {net_pnl:+,.2f}</code></b>\n"
        )

        if costs:
            total_cost = costs.get("total", 0)
            text += (
                f"\n<i>Costs: Rs {total_cost:,.2f}</i>\n"
                f"<i>(Brokerage {costs.get('brokerage', 0):.1f} + "
                f"STT {costs.get('stt', 0):.1f} + "
                f"GST {costs.get('gst', 0):.1f} + others)</i>\n"
            )

        text += f"\n\U0001f4b0 Portfolio: <b><code>Rs {equity:,.0f}</code></b>"
        self.send_message(text)

    def alert_risk_breach(self, risk_info: Dict):
        """Alert when a risk limit is breached."""
        text = (
            "\u26a0\ufe0f <b>RISK ALERT</b>\n"
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

        wr_line = f"Win Rate: <code>{wins/trades*100:.0f}%</code>\n" if trades > 0 else ""

        text = (
            f"{pnl_emoji} <b>DAILY SUMMARY</b>\n"
            f"<code>{datetime.now().strftime('%d %b %Y')}</code>\n"
            f"Gross P&L: <code>Rs {gross_pnl:+,.0f}</code>\n"
        )
        if total_costs > 0:
            text += f"Costs: <code>Rs {total_costs:,.0f}</code>\n"
        text += (
            f"<b>Net P&L: <code>Rs {pnl:+,.0f}</code></b>\n"
            f"Trades: <code>{trades}</code> (Won: {wins})\n"
            f"{wr_line}"
        )
        if guardrail_audit:
            text += f"Intraday Guardrail: <code>{guardrail_audit}</code>\n"
        text += f"\n\U0001f4b0 Portfolio: <b><code>Rs {equity:,.0f}</code></b>"
        self.send_message(text)

    def alert_system_start(self):
        """Alert when system starts."""
        text = (
            "\U0001f680 <b>PROMETHEUS ONLINE</b>\n"
            f"<code>{datetime.now().strftime('%d %b %Y  %H:%M')}</code>\n"
            "System is monitoring markets.\n\n"
            "/scan  |  /status  |  /pnl\n"
            "/positions  |  /regime  |  /help"
        )
        self.send_message(text)

    def alert_system_error(self, error: str):
        """Alert on critical system error (non-blocking)."""
        text = (
            "\U0001f525 <b>SYSTEM ERROR</b>\n"
            f"<code>{error[:300]}</code>"
        )
        self.send_message_async(text)
