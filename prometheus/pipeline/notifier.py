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
        """Generate Kite-searchable contract name.
        
        Kite search format: BANKNIFTY JUN 45000 CE
        This is what users paste into Kite's search bar.
        """
        if not signal.strike or signal.strike <= 0:
            return ''
        
        # Determine underlying name for Kite
        sym = signal.symbol.upper()
        INDEX_MAP = {
            'SENSEX': 'SENSEX',
            'NIFTY 50': 'NIFTY',
            'NIFTY BANK': 'BANKNIFTY',
            'NIFTY FIN SERVICE': 'FINNIFTY',
            'NIFTY MIDCAP SELECT': 'MIDCPNIFTY',
        }
        underlying = INDEX_MAP.get(signal.symbol, sym)
        
        # Month from expiry date
        month_str = ''
        if signal.expiry:
            try:
                from datetime import datetime as dt
                d = dt.strptime(signal.expiry, '%Y-%m-%d')
                month_str = d.strftime('%b').upper()  # JUN, JUL etc.
            except Exception:
                month_str = ''
        
        strike_str = str(int(signal.strike)) if signal.strike == int(signal.strike) else str(signal.strike)
        otype = signal.option_type or 'CE'
        
        if month_str:
            return f"{underlying} {month_str} {strike_str} {otype}"
        else:
            return f"{underlying} {strike_str} {otype}"
    
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
        lines.append(f'Entry: <code>Rs {signal.entry_price:,.2f}</code>')
        lines.append(f'SL: <code>Rs {signal.stop_loss:,.2f}</code>')
        lines.append(f'Target: <code>Rs {signal.target:,.2f}</code>')
        lines.append(f'RR: <code>1:{signal.risk_reward:.1f}</code>')
        
        # Try to get live premium from Angel One
        premium_str = self._get_live_premium(signal)
        if premium_str:
            lines.append(f'')
            lines.append(f'\U0001f4b0 Premium: <code>{premium_str}</code>')
        
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
        """Send message via telegram (or log if not available)."""
        if self._telegram:
            try:
                self._telegram.send_message(text)
            except Exception as e:
                logger.error(f"Notifier: telegram send failed: {e}")
        logger.info(f"Notifier: {text[:200]}")
