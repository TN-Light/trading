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
    
    def notify_signal_alert(self, signal: ExecutableSignal):
        """Send a detailed signal alert (for primary account)."""
        direction = 'BULLISH' if signal.direction == 'bullish' else 'BEARISH'
        emoji = '\U0001f7e2' if signal.direction == 'bullish' else '\U0001f534'
        
        lines = [
            f'{emoji} <b>SIGNAL: {signal.symbol}</b>',
            f'Direction: {direction}',
            f'Action: <code>{signal.action}</code>',
            f'',
            f'Entry: <code>Rs {signal.entry_price:,.2f}</code>',
            f'SL: <code>Rs {signal.stop_loss:,.2f}</code>',
            f'Target: <code>Rs {signal.target:,.2f}</code>',
            f'RR: <code>1:{signal.risk_reward:.1f}</code>',
        ]
        
        if signal.instrument:
            lines.append(f'Contract: <code>{signal.instrument}</code>')
        elif signal.strike > 0:
            lines.append(
                f'Strike: <code>{int(signal.strike)} {signal.option_type}</code>'
            )
        
        if signal.regime:
            lines.append(f'Regime: <code>{signal.regime}</code>')
        if signal.reasons:
            lines.append(f'Signals: <code>{", ".join(signal.reasons)}</code>')
        
        lines.append(f'Confidence: <code>{signal.confidence:.0%}</code>')
        
        self._send('\n'.join(lines))
    
    def notify_execution_result(self, signal: ExecutableSignal, position, error: str = ''):
        """Notify about trade execution or rejection."""
        if position:
            self._send(
                f'\u2705 <b>PAPER TRADE OPENED</b>\n'
                f'{signal.symbol} {signal.action}\n'
                f'Position: <code>{getattr(position, "position_id", "unknown")}</code>'
            )
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
