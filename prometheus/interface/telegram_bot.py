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
                # DEBUG (not INFO) — _try_connect is also invoked by the
                # transport watchdog health probe (every ~3 min) and by
                # _init_bot's strategy loops. Promoting to INFO here caused
                # redundant "Telegram bot connected" log spam every 3 min
                # even when no actual reconnect happened. The "Telegram
                # connected via <strategy>" INFO log in _init_bot is the
                # operator-visible proof of an established connection.
                logger.debug(
                    f"Telegram probe ok: @{bot_info.get('username', 'unknown')} "
                    f"base={base_url}"
                )
                return True
        except Exception as exc:
            logger.debug(f"Telegram probe failed on {base_url}: {exc}")
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
                        # Item 3 (2026-07-25 audit follow-up): one-site
                        # DEBUG log per successful send so future
                        # instrumented audits can rely on a stable
                        # success-trace without grep-ing through call
                        # sites. DEBUG level (not INFO) — a paper bot
                        # sends hundreds of fills/skips/stats per
                        # session; promoting to INFO would drown the
                        # log. Operators who need it visible can set
                        # ``logger.setLevel("DEBUG")`` on the
                        # ``prometheus.interface.telegram_bot`` logger.
                        logger.debug(
                            f"Telegram send ok chat_id={self.chat_id} "
                            f"len={len(text)} base={base_url}"
                        )
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

    def alert_new_signal(self, signal: Dict, source: str = ""):
        """Send a compact mobile-friendly trading signal alert.

        Args:
            signal: Signal data dict.
            source: Origin tag — 'scan' (manual /scan), 'auto' (auto-scan loop),
                    'multi' (multi-account dispatch), or '' (legacy/untagged).
        """
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
        trade_mode = str(signal.get("trade_mode") or signal.get("timeframe") or "").lower()
        hold_line = ""
        if trade_mode == "swing":
            hold_line = "\nHold: <code>overnight carry</code>"
        elif trade_mode == "intraday":
            hold_line = "\nHold: <code>same-day square-off</code>"

        if action == "HOLD":
            return

        # ── DEDICATED SPREAD ALERT FORMATTING (BARBELL DUAL-REGIME) ──
        if signal.get("strategy_type") == "credit_spread" or "SPREAD" in action:
            spread_type = signal.get("spread_type", action)
            net_credit = float(signal.get("net_credit", entry) or 0)
            target_decay = float(signal.get("target_decay_price", target) or 0)
            hard_sl = float(signal.get("hard_sl_price", sl) or 0)
            margin_req = float(signal.get("margin_required", 35000) or 35000)
            legs = signal.get("legs", [])
            
            legs_text = ""
            for leg in legs:
                leg_act = leg.get("action", "BUY")
                leg_sym = leg.get("tradingsymbol", "")
                leg_prem = float(leg.get("premium", 0.0) or 0)
                leg_tag = "(Hedge Margin Reducer)" if leg.get("is_hedge") else "(Short Theta Collector)"
                legs_text += f"• <b>{leg_act}</b> <code>{leg_sym}</code> (~Rs {leg_prem:.1f}) {leg_tag}\n"

            # Source tag for alert segregation
            if source == "scan":
                source_tag = "  📡 <i>/scan</i>"
            elif source == "auto":
                source_tag = "  ⚡ <i>auto</i>"
            elif source == "multi":
                source_tag = "  ⚡ <i>multi</i>"
            else:
                source_tag = ""

            text = (
                f"🛡️ <b>NEW BARBELL SIGNAL: {spread_type}</b>{source_tag}\n"
                f"<b>Underlying:</b> <code>{symbol}</code>\n\n"
                f"<b>Legs (Copy & Search in Kite/Angel):</b>\n{legs_text}\n"
                f"<b>Net Credit:</b> Rs {net_credit:,.1f}/share\n"
                f"<b>Target Exit (70% Decay):</b> Rs {target_decay:,.1f}\n"
                f"<b>Hard Stop Loss (1.5x):</b> Rs {hard_sl:,.1f}\n"
                f"<b>Est. Margin Required:</b> Rs {margin_req:,.0f}/lot\n"
                f"<b>Hold:</b> <code>same-day theta decay</code>\n"
            )
            self.send_message(text)
            return

        # Generate user-friendly contract name if expiry and strike are present
        friendly_contract = ""
        if expiry and strike and option_type:
            try:
                from datetime import datetime as dt
                from datetime import timedelta
                if isinstance(expiry, str):
                    d = dt.strptime(expiry[:10], "%Y-%m-%d")
                else:
                    d = expiry
                strike_str = str(int(float(strike)))
                mon = d.strftime("%b").upper()
                
                next_week = d + timedelta(days=7)
                is_monthly = next_week.month != d.month
                
                if is_monthly:
                    friendly_contract = f"{symbol} {mon} {strike_str} {option_type}"
                else:
                    day = d.day
                    friendly_contract = f"{symbol} {day} {mon} {strike_str} {option_type}"
            except Exception as e:
                logger.error(f"Error formatting friendly contract name: {e}")

        # Account details if applicable
        account_header = ""
        account_label = signal.get("account_label", "")
        if account_label:
            capital_val = float(signal.get("account_capital", 0) or 0)
            cap_str = f" | Capital: Rs {capital_val:,.0f}" if capital_val > 0 else ""
            account_header = f"📊 <b>ACCOUNT: {account_label.upper()}</b>{cap_str}\n"

        # Trade sizing details
        qty = int(signal.get("quantity", 0) or 0)
        lots = int(signal.get("lots", 0) or 0)
        lot_size = int(signal.get("lot_size", 0) or 0)
        if lots > 0:
            qty_part = f" ({qty} Qty)" if qty > 0 else ""
            sizing_line = f"Sizing: <code>{lots} Lots</code>{qty_part}\n"
        elif qty > 0:
            sizing_line = f"Sizing: <code>{qty} Qty</code>\n"
        else:
            sizing_line = ""

        # Investment / Margin required
        lot_cost = float(signal.get("lot_cost", 0) or 0)
        if lot_cost > 0 and lots > 0:
            total_cost = lot_cost * lots
            cost_line = f"Margin Required: <code>Rs {total_cost:,.0f}</code> (Rs {lot_cost:,.0f}/lot)\n"
        elif lot_cost > 0:
            cost_line = f"Margin Required: <code>Rs {lot_cost:,.0f}</code>\n"
        else:
            cost_line = ""

        emoji = "\U0001f7e2" if "CE" in action else "\U0001f534"
        direction = "BULLISH" if "CE" in action else "BEARISH"

        quality, wr = REGIME_QUALITY.get(regime, ("???", ""))
        caution = ""
        if quality == "WEAK":
            caution = "\n\u26a0\ufe0f Low-confidence regime (26% WR)"
        elif quality == "LOW":
            caution = "\n\u26a0\ufe0f Volatile regime — lower conviction"

        contract_line = ""
        tradingsymbol = signal.get("tradingsymbol", "")
        tsym_part = f" (<code>{tradingsymbol}</code>)" if tradingsymbol else ""
        if friendly_contract:
            contract_line = f"Contract: <code>{friendly_contract}</code>{tsym_part}\n"
        elif instrument:
            contract_line = f"Contract: <code>{instrument}</code>{tsym_part}\n"
        elif strike and option_type:
            exp = f" {expiry}" if expiry else ""
            contract_line = f"Contract: <code>{symbol}{exp} {int(float(strike))}{option_type}</code>{tsym_part}\n"

        # Source tag for alert segregation
        if source == "scan":
            source_tag = "  \U0001f4e1 <i>/scan</i>"
        elif source == "auto":
            source_tag = "  \u26a1 <i>auto</i>"
        elif source == "multi":
            source_tag = "  \u26a1 <i>multi</i>"
        else:
            source_tag = ""

        text = (
            f"{emoji} <b>NEW TRADING SIGNAL</b>{source_tag}\n"
            f"{account_header}\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Action:</b> {action} {contract_line.replace('Contract: ', '')}\n"
            f"<b>Entry Price:</b> Rs {entry:,.1f}\n"
            f"<b>Stop Loss:</b> Rs {sl:,.1f}\n"
            f"<b>Target:</b> Rs {target:,.1f}\n\n"
            f"<i>{sizing_line.replace('Sizing: ', 'Quantity: ')}</i>"
            f"<i>{cost_line.replace('Margin Required: ', 'Est. Capital: ')}</i>"
            f"{hold_line}\n"
        )
        if reasoning:
            text += f"\n<i>Note: {reasoning[:150]}</i>"

        self.send_message(text)

    def alert_scanner_summary(self, scan_results: List[Dict]):
        """Send multi-index scanner results in a compact, mobile-friendly format.

        Freshness guard (Fix B, 2026-08-17):
        Each scan_result may carry ``_signal_generated_at`` (unix ts, set in
        ``main._scan_one_cmd`` / ``_scan_intra_cmd``). If the delay between
        signal generation and this summary send exceeds
        ``STALE_THRESHOLD_SEC`` (default 90s), the row is tagged with a
        ``⚠️ STALE +Xs`` badge in the displayed list, and is EXCLUDED from
        the per-signal ``alert_new_signal`` follow-up call — a /scan is an
        inquiry, not an order placement, but operators have manually traded
        off stale /scan alerts. Flagging the staleness prevents the "signal
        said 268, Kite already shows 256" class of mistakes.
        """
        STALE_THRESHOLD_SEC = 90
        now_ts = time.time()

        def _age(r):
            ts = r.get("_signal_generated_at")
            if not ts:
                return None
            return max(0, int(now_ts - ts))

        def _is_stale(r):
            age = _age(r)
            return age is not None and age > STALE_THRESHOLD_SEC

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

        stale_count = sum(1 for r in results if _is_stale(r))
        header_lines = [
            "\U0001f50d <b>MARKET SCAN</b>",
            f"<code>{datetime.now().strftime('%d %b %Y  %H:%M')}</code>",
            "<i>Showing top-ranked actionable signals only.</i>",
            "",
        ]
        if stale_count:
            header_lines.insert(
                2,
                f"\u26a0\ufe0f <b>{stale_count} stale signal(s) (>{STALE_THRESHOLD_SEC}s old)</b> "
                f"— premium may have moved. Do NOT enter without re-checking live LTP.",
            )
        lines = list(header_lines)

        def _format_row(r):
            action = r.get("action", "HOLD")
            symbol = r.get("symbol", "")
            regime = r.get("regime", "unknown")
            adj_conf = r.get("adj_confidence", 0)
            sig_count = r.get("signal_count", 0)
            tf = r.get("timeframe", "15minute")
            tf_clean = tf.replace("intraday ", "").replace("minute", "m").replace("day", "D")

            direction = "CE Buy" if "CE" in action else "PE Buy" if "PE" in action else "HOLD"
            d_emoji = "🟢" if "CE" in action else "🔴" if "PE" in action else "⚪"

            sig_only = " (sig only)" if not r.get("executable", True) else ""
            stale_tag = f" \u26a0\ufe0f STALE +{_age(r)}s" if _is_stale(r) else ""

            return (
                f"{d_emoji} <b>{symbol} ({tf_clean})</b>: {direction} | Conf {adj_conf:.0%} | {sig_count}/10 sigs | {regime.upper()}{sig_only}{stale_tag}"
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
        # Fix B: do NOT trigger the per-signal follow-up alert for stale
        # signals — the displayed row already carries the STALE badge. The
        # per-signal alert would otherwise look identical to a real-time
        # auto-scan signal and could be manually traded against a stale price.
        actionable_fresh = [r for r in actionable if not _is_stale(r)]
        if actionable_fresh:
            lines.append(f"\U0001f525 <b>{len(actionable_fresh)} actionable signal(s)</b> above 50%")
        elif actionable:
            lines.append(
                f"\u23f8 {len(actionable)} signal(s) above 50% but ALL stale (>{STALE_THRESHOLD_SEC}s) — skipped per-signal alerts"
            )
        else:
            lines.append("\u23f8 No signals above 50% threshold")

        self.send_message("\n".join(lines))

        # Fire per-signal alerts only for fresh actionable rows.
        for r in actionable_fresh[:3]:
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
                "trade_mode": r.get("timeframe", ""),
            }, source="scan")

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
        """Alert when a stop loss is hit (phase1 — actual loss to original SL)."""
        pnl = trade_info.get("pnl", 0)
        text = (
            "\U0001f6d1 <b>STOP LOSS HIT</b>\n"
            f"{trade_info.get('symbol', '')}\n"
            f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
            f"P&L: <code>Rs {pnl:+,.0f}</code>"
        )
        self.send_message(text)

    def alert_trailing_lock_hit(self, trade_info: Dict, phase: str = ""):
        """Alert when the 5-stage trailing stop locks in profit (phase2/phase3).

        These exits hit a trailing SL — but at a price ABOVE entry for phase3
        (lock 70%) and near entry for phase2 (lock 20%). Reporting them as
        'STOP LOSS HIT' was mislabelling profit-locks as losses, making a
        profitable exit look like a SL hit.
        """
        pnl = trade_info.get("pnl", 0)
        tag = f" ({phase})" if phase else ""
        if pnl >= 0:
            text = (
                "\U0001f4c8 <b>TRAILING LOCKED</b>" + tag + "\n"
                f"{trade_info.get('symbol', '')}\n"
                f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
                f"P&L: <code>Rs {pnl:+,.0f}</code>"
            )
        else:
            text = (
                "\U0001f6d1 <b>TRAILING EXIT</b>" + tag + "\n"
                f"{trade_info.get('symbol', '')}\n"
                f"Exit: <code>Rs {trade_info.get('exit_price', 0):,.2f}</code>\n"
                f"P&L: <code>Rs {pnl:+,.0f}</code>"
            )
        self.send_message(text)

    def alert_adverse_exit(self, symbol, instrument, entry, exit_price, pnl, reason="SuperTrend reversal"):
        msg = (
            f"\U000026A0 <b>ADVERSE EXIT</b>\n"
            f"Symbol: {symbol}\n"
            f"Instrument: {instrument}\n"
            f"Entry: {entry:.2f} → Exit: {exit_price:.2f}\n"
            f"PnL: Rs {pnl:+.2f}\n"
            f"Reason: {reason}\n"
            f"\U0001f504 SuperTrend flipped against position"
        )
        self.send_message(msg)

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
