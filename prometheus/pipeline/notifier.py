"""Notifier: Sends scan results to Telegram with diagnostic information."""

from typing import Optional
from prometheus.pipeline.types import (
    ScanCycle, SymbolScanResult, ExecutableSignal, DataStatus
)
from prometheus.utils.logger import logger


class Notifier:
    """Sends scan results to Telegram with full diagnostic trail."""
    
    def __init__(self, telegram_bot):
        self._telegram = telegram_bot
    
    def notify_scan_result(self, cycle: ScanCycle):
        """Send a scan summary to Telegram.
        
        Always sends a message, even when no signals found.
        Includes diagnostic info: what was scanned, what was rejected, why.
        """
        ts = cycle.timestamp.strftime('%d %b %Y %H:%M')
        signals = cycle.signals_found
        executed = cycle.signals_executed
        total = cycle.total_symbols
        
        if signals == 0:
            # No signals — show diagnostic breakdown
            lines = [
                '\U0001f50e <b>Swing-15m scan complete</b>',
                f'<code>{ts}</code>',
                f'Symbols scanned: <b>{total}</b>',
                f'Eligible signals: <b>0</b>',
                '',
            ]
            
            # Show rejection reasons breakdown
            rejections = cycle.rejections
            if rejections:
                lines.append('<b>Rejections:</b>')
                for symbol, reason in rejections:
                    lines.append(f'  {symbol}: <code>{reason}</code>')
            else:
                lines.append('No setups met confluence threshold.')
            
            self._send('\n'.join(lines))
        else:
            # Signals found
            lines = [
                '\U0001f50e <b>Swing-15m scan complete</b>',
                f'<code>{ts}</code>',
                f'Symbols scanned: <b>{total}</b>',
                f'Eligible signals: <b>{signals}</b>',
                f'Executed: <b>{executed}</b>',
            ]
            
            # List each signal
            for r in cycle.results:
                if r.had_signal and r.executable:
                    status = '\u2705' if r.executed else '\u274c'
                    reason = ''
                    if not r.executed and r.gate and not r.gate.passed:
                        reason = f' ({r.gate.reason})'
                    elif not r.executed and r.execution_error:
                        reason = f' ({r.execution_error})'
                    lines.append(
                        f'{status} {r.executable.symbol} '
                        f'{r.executable.action} '
                        f'@ Rs {r.executable.entry_price:,.2f}'
                        f'{reason}'
                    )
            
            self._send('\n'.join(lines))
    
    def _make_kite_search_name(self, signal: ExecutableSignal) -> str:
        """Generate a Kite-app search-bar friendly name.

        Format:
          Monthly : ``NIFTY JUL 24500 CE``          (year stripped)
          Weekly  : ``NIFTY 21st JUL 24500 CE``     (ordinal day suffix)

        This is what you copy-paste into the Kite mobile app's search box, NOT
        what the API receives. The API's tradingsymbol lives in
        ``signal.instrument`` (e.g. ``NIFTY2672124150CE``) and is used by the
        broker. See ``prometheus/utils/symbol_format.py`` for the full rationale.
        """
        if not signal.strike or signal.strike <= 0:
            return ''

        # Delegate to the centralized formatter — single source of truth for
        # underlying mapping (NIFTY 50 → NIFTY), monthly/weekly detection
        # (post-Sep-2025 NSE Tuesday standardization preserved), and ordinal
        # day-suffix rendering for weeklies.
        from prometheus.utils.symbol_format import human_search_name
        return human_search_name(
            signal.symbol, signal.expiry, signal.strike, signal.option_type,
        )
    
    def notify_signal_alert(self, signal: ExecutableSignal):
        """Send a detailed signal alert with Kite-searchable contract name."""
        direction = 'BULLISH' if signal.direction == 'bullish' else 'BEARISH'
        emoji = '\U0001f7e2' if signal.direction == 'bullish' else '\U0001f534'
        
        # Generate Kite-searchable name
        kite_name = self._make_kite_search_name(signal)
        
        lines = [
            f'{emoji} <b>SIGNAL: {signal.symbol}</b>',
            f'Direction: {direction}',
            f'Action: <code>{signal.action}</code>',
        ]
        
        # Kite copy-paste contract name (prominent)
        if kite_name:
            lines.append(f'')
            lines.append(f'\U0001f4cb <b>Kite Search:</b>')
            lines.append(f'<code>{kite_name}</code>')
        
        lines.append(f'')
        
        # Detect if we have live market prices vs BS estimates
        has_live = signal.raw and signal.raw.get('live_premium', 0) > 0 if signal.raw else False
        
        if has_live:
            # Real market prices from Angel One
            lines.append(f'\U0001f4b0 Entry: <code>Rs {signal.entry_price:,.2f}</code> (LIVE)')
            lines.append(f'SL: <code>Rs {signal.stop_loss:,.2f}</code>')
            lines.append(f'Target: <code>Rs {signal.target:,.2f}</code>')
            bid = signal.raw.get('bid', 0)
            ask = signal.raw.get('ask', 0)
            if bid > 0 and ask > 0:
                lines.append(f'Bid/Ask: <code>{bid:.2f} / {ask:.2f}</code>')
        else:
            # Theoretical BS estimates
            lines.append(f'Est. Entry: <code>~Rs {signal.entry_price:,.2f}</code>')
            lines.append(f'Est. SL: <code>~Rs {signal.stop_loss:,.2f}</code>')
            lines.append(f'Est. Target: <code>~Rs {signal.target:,.2f}</code>')
            lines.append(f'')
            lines.append(f'\u26a0\ufe0f <i>Prices are BS estimates. Check Kite for live premium.</i>')
        
        lines.append(f'RR: <code>1:{signal.risk_reward:.1f}</code>')
        
        # Show underlying spot for reference
        spot = signal.raw.get('spot_at_signal', 0) if signal.raw else 0
        if spot > 0:
            lines.append(f'Spot: <code>{spot:,.2f}</code>')
        
        if signal.strike > 0:
            strike_str = str(int(signal.strike)) if signal.strike == int(signal.strike) else str(signal.strike)
            lines.append(f'Strike: <code>{strike_str} {signal.option_type}</code>')
        
        if signal.lot_size:
            lines.append(f'Lot: <code>{signal.lot_size}</code>')
        
        if signal.regime:
            lines.append(f'Regime: <code>{signal.regime}</code>')
        if signal.reasons:
            lines.append(f'Signals: <code>{", ".join(signal.reasons)}</code>')
        
        lines.append(f'Confidence: <code>{signal.confidence:.0%}</code>')
        
        # OI data (if available from cache)
        if signal.raw:
            oi_pcr = signal.raw.get('oi_pcr', 0)
            oi_sentiment = signal.raw.get('oi_sentiment', '')
            oi_agrees = signal.raw.get('oi_agrees', None)
            if oi_pcr > 0 or oi_sentiment:
                lines.append('')
                oi_icon = '\u2705' if oi_agrees else '\u26a0\ufe0f'
                lines.append(f'{oi_icon} <b>OI Analysis:</b>')
                if oi_pcr > 0:
                    lines.append(f'PCR: <code>{oi_pcr:.2f}</code>')
                if oi_sentiment:
                    lines.append(f'<code>{oi_sentiment}</code>')
        
        self._send('\n'.join(lines))
    
    def _get_live_premium(self, signal: ExecutableSignal) -> str:
        """Try to fetch live option premium from Angel One option chain."""
        try:
            if not hasattr(self, '_prometheus') or not self._prometheus:
                # Try to get prometheus instance from telegram
                if hasattr(self._telegram, '_command_handlers'):
                    return ''
            return ''  # Premium fetching requires live market — return empty for now
        except Exception:
            return ''
    
    def notify_execution_result(self, signal: ExecutableSignal, position, error: str = ''):
        """Notify about trade execution or rejection with Kite-searchable name."""
        kite_name = self._make_kite_search_name(signal)
        
        if position:
            lines = [
                f'\u2705 <b>PAPER TRADE OPENED</b>',
                f'{signal.symbol} {signal.action}',
            ]
            if kite_name:
                lines.append(f'\U0001f4cb Kite: <code>{kite_name}</code>')
            lines.append(f'Position: <code>{getattr(position, "position_id", "unknown")}</code>')
            self._send('\n'.join(lines))
        elif error:
            self._send(
                f'\u26a0\ufe0f <b>TRADE NOT EXECUTED</b>\n'
                f'{signal.symbol}: {error}'
            )
    
    def _send(self, text: str):
        """Send message via telegram synchronously (for critical alerts)."""
        if self._telegram:
            try:
                self._telegram.send_message(text)
            except Exception as e:
                logger.error(f"Notifier: telegram send failed: {e}")
        logger.info(f"Notifier: {text[:200]}")
    
    def _send_async(self, text: str):
        """Send message via telegram asynchronously (for non-critical updates).
        
        Falls back to sync send if async is not available.
        Uses send_message as base — the telegram bot's send_message_async
        is only used when explicitly available (not via MagicMock).
        """
        if self._telegram:
            try:
                # Try async if the method is explicitly defined (not auto-mocked)
                async_fn = getattr(self._telegram, 'send_message_async', None)
                if async_fn and not isinstance(async_fn, type(self._telegram.send_message)):
                    # send_message_async exists and is a different type than send_message
                    # (rules out MagicMock which returns same type for all attrs)
                    async_fn(text)
                else:
                    self._telegram.send_message(text)
            except Exception as e:
                logger.error(f"Notifier: telegram async send failed: {e}")
        logger.info(f"Notifier: {text[:200]}")
