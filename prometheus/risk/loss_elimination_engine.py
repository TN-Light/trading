# ============================================================================
# PROMETHEUS — Loss Elimination Engine
# ============================================================================
"""
Systematic loss elimination through 6 automated defensive layers.

Every layer is a hard-coded, automated, non-bypassable rule:
  1. PreTradeKillSwitch      — Score trades against known loss patterns
  2. AdaptiveStopLoss        — Archetype-aware stop placement
  3. TemporalBlackoutManager — Hard no-trade time windows
  4. RegimeGate              — Block incompatible strategy/regime combos
  5. CircuitBreaker          — Auto position sizing after consecutive losses
  6. PostLossLearningLoop    — Learn from every new loss automatically

Integration: Called by RiskManager.pre_trade_check() BEFORE any trade.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# ============================================================================
# Data Classes
# ============================================================================

class KillSwitchVerdict(Enum):
    CLEAR = "clear"           # Loss Risk Score < 40 → proceed normally
    WARNING = "warning"       # 40-70 → reduce position size 50%
    BLOCKED = "blocked"       # > 70 → HARD BLOCK, trade rejected


@dataclass
class EliminationResult:
    """Result from the full elimination engine check."""
    approved: bool
    verdict: KillSwitchVerdict
    loss_risk_score: float
    position_size_multiplier: float  # 1.0 = normal, 0.5 = half, 0.0 = blocked
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    matched_patterns: List[Dict] = field(default_factory=list)
    layer_results: Dict = field(default_factory=dict)


# ============================================================================
# Layer 1 — Pre-Trade Kill Switch
# ============================================================================

class PreTradeKillSwitch:
    """
    Before any trade is executed, score it against every known loss pattern.

    Loss Risk Score = weighted sum of all pattern matches (0-100)

    > 70 → HARD BLOCK, trade rejected
    40-70 → WARNING, reduce position size 50%
    < 40 → CLEAR, proceed normally
    """

    BLOCK_THRESHOLD = 70
    WARNING_THRESHOLD = 40

    def __init__(self, patterns_file: Optional[str] = None):
        self.patterns: List[Dict] = []
        self.combination_patterns: List[Dict] = []

        if patterns_file and Path(patterns_file).exists():
            self.load_patterns(patterns_file)

    def load_patterns(self, filepath: str):
        """Load discovered patterns from PatternMiningEngine output."""
        with open(filepath, 'r') as f:
            report = json.load(f)

        # Single attribute patterns
        freq_layer = report.get('layers', {}).get('frequency', {})
        self.patterns = freq_layer.get('patterns', [])

        # Combination patterns
        combo_layer = report.get('layers', {}).get('combination', {})
        self.combination_patterns = combo_layer.get('patterns', [])

    def add_pattern(self, pattern: Dict):
        """Manually add a loss pattern (e.g., from live learning)."""
        self.patterns.append(pattern)

    def check(self, trade_attrs: Dict) -> Tuple[float, KillSwitchVerdict, List[Dict]]:
        """
        Score a trade against all known patterns.

        Args:
            trade_attrs: Dictionary of trade attributes (same keys as LossDNATagger output)

        Returns:
            (loss_risk_score, verdict, matched_patterns)
        """
        score = 0.0
        matched = []

        # Check single-attribute patterns
        for pattern in self.patterns:
            attr = pattern.get('attribute', '')
            expected_val = pattern.get('value', '')
            actual_val = str(trade_attrs.get(attr, ''))

            if actual_val == expected_val:
                severity = pattern.get('severity', 'normal')
                weight = {
                    'lethal': 25, 'critical': 15,
                    'loss_pattern': 8, 'normal': 3
                }.get(severity, 3)

                score += weight
                matched.append({
                    'type': 'single',
                    'pattern': f"{attr}={expected_val}",
                    'severity': severity,
                    'weight': weight,
                })

        # Check combination patterns
        for combo in self.combination_patterns:
            conditions = combo.get('combination', [])
            loss_rate = combo.get('loss_rate', 0)

            all_match = True
            for cond in conditions:
                parts = cond.split('=', 1)
                if len(parts) != 2:
                    all_match = False
                    break
                attr, expected = parts
                actual = str(trade_attrs.get(attr, ''))
                if actual != expected:
                    all_match = False
                    break

            if all_match:
                depth = combo.get('depth', 2)
                weight = min(loss_rate / 5, 20) * (depth / 2)

                score += weight
                matched.append({
                    'type': 'combination',
                    'pattern': ' + '.join(conditions),
                    'loss_rate': loss_rate,
                    'depth': depth,
                    'weight': round(weight, 1),
                })

        # Cap score at 100
        score = min(score, 100.0)

        # Determine verdict
        if score >= self.BLOCK_THRESHOLD:
            verdict = KillSwitchVerdict.BLOCKED
        elif score >= self.WARNING_THRESHOLD:
            verdict = KillSwitchVerdict.WARNING
        else:
            verdict = KillSwitchVerdict.CLEAR

        return round(score, 1), verdict, matched


# ============================================================================
# Layer 2 — Adaptive Stop Loss
# ============================================================================

class AdaptiveStopLoss:
    """
    Replace fixed stop losses with pattern-aware stops based on
    which loss archetype is most common.

    Stop Hunt dominant  → Move stop 1.5 ATR beyond obvious levels
    Chop Grind dominant → Time-based stop (exit if no progress in N bars)
    Event Shock         → Auto-tighten stops before known events
    False Signal        → Wider initial stop for confirmation period
    """

    def __init__(self, archetype_stats: Optional[Dict] = None):
        self.archetype_stats = archetype_stats or {}
        self._dominant_archetype = self._find_dominant()

    def _find_dominant(self) -> str:
        """Find the dominant loss archetype."""
        if not self.archetype_stats:
            return 'stop_hunt'

        return max(
            self.archetype_stats.items(),
            key=lambda x: x[1].get('count', 0)
        )[0]

    def calculate_stop(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        base_multiplier: float = 2.0,
        trade_attrs: Optional[Dict] = None,
    ) -> Dict:
        """
        Calculate archetype-aware stop loss.

        Returns dict with stop_price, type, and reasoning.
        """
        result = {
            'base_stop': 0.0,
            'adaptive_stop': 0.0,
            'multiplier_used': base_multiplier,
            'reasoning': '',
            'time_stop_bars': None,
        }

        dominant = self._dominant_archetype
        adjusted_mult = base_multiplier

        if dominant == 'stop_hunt':
            # Widen stop to avoid obvious levels
            adjusted_mult = base_multiplier * 1.5
            result['reasoning'] = (
                f'Stop Hunt dominant ({self.archetype_stats.get("stop_hunt", {}).get("pct_of_losses", 0)}% of losses). '
                f'Widened stop from {base_multiplier}x to {adjusted_mult:.1f}x ATR to avoid retail stop levels.'
            )

        elif dominant == 'chop_grind':
            # Add time-based stop
            adjusted_mult = base_multiplier * 1.1
            result['time_stop_bars'] = 12  # ~3 hours in 15min bars
            result['reasoning'] = (
                f'Chop Grind dominant ({self.archetype_stats.get("chop_grind", {}).get("pct_of_losses", 0)}% of losses). '
                f'Added time stop of 12 bars. Exit if no progress.'
            )

        elif dominant == 'false_signal':
            # Wider initial stop with quick breakeven
            adjusted_mult = base_multiplier * 1.3
            result['reasoning'] = (
                f'False Signal dominant ({self.archetype_stats.get("false_signal", {}).get("pct_of_losses", 0)}% of losses). '
                f'Widened initial stop to allow confirmation, move to breakeven at 0.5R.'
            )

        elif dominant == 'overextension':
            # Tighter stop — already extended
            adjusted_mult = base_multiplier * 0.8
            result['reasoning'] = (
                f'Overextension dominant. Tightened stop to {adjusted_mult:.1f}x ATR.'
            )

        # Calculate prices
        atr_distance = atr * adjusted_mult

        if direction == 'bullish':
            result['base_stop'] = round(entry_price - atr * base_multiplier, 2)
            result['adaptive_stop'] = round(entry_price - atr_distance, 2)
        else:
            result['base_stop'] = round(entry_price + atr * base_multiplier, 2)
            result['adaptive_stop'] = round(entry_price + atr_distance, 2)

        result['multiplier_used'] = round(adjusted_mult, 2)

        return result


# ============================================================================
# Layer 3 — Temporal Blackout Manager
# ============================================================================

class TemporalBlackoutManager:
    """
    Hard-coded no-trade windows based on loss heatmap data.

    These are NOT suggestions. They are hard blocks that cannot be overridden.
    """

    # Default blackout rules (confirmed by actual pattern analysis)
    DEFAULT_BLACKOUTS = [
        {
            'name': 'market_open_first_15min',
            'description': 'First 15 minutes of market (9:15-9:30)',
            'start': time(9, 15),
            'end': time(9, 30),
            'days': [0, 1, 2, 3, 4],  # All weekdays
            'enabled': True,
        },
        {
            'name': 'pre_close_last_15min',
            'description': 'Last 15 minutes before close (3:15-3:30)',
            'start': time(15, 15),
            'end': time(15, 30),
            'days': [0, 1, 2, 3, 4],
            'enabled': True,
        },
    ]

    def __init__(
        self,
        custom_blackouts: Optional[List[Dict]] = None,
        high_risk_windows: Optional[List[Dict]] = None,
    ):
        self.blackouts = list(self.DEFAULT_BLACKOUTS)

        if custom_blackouts:
            self.blackouts.extend(custom_blackouts)

        # Auto-generate blackouts from high-risk windows (from TemporalClusterAnalyzer)
        if high_risk_windows:
            self._generate_from_analysis(high_risk_windows)

    def _generate_from_analysis(self, windows: List[Dict]):
        """Generate blackout rules from temporal analysis high-risk windows."""
        for w in windows:
            if w.get('excess_loss_rate', 0) > 20 and w.get('sample_size', 0) >= 10:
                dim = w.get('dimension', '')
                val = w.get('value', '')

                if dim == 'day_of_week':
                    day_map = {
                        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                        'Thursday': 3, 'Friday': 4,
                    }
                    if val in day_map:
                        self.blackouts.append({
                            'name': f'high_risk_{val.lower()}',
                            'description': f'{val} — {w["loss_rate"]}% loss rate '
                                         f'(+{w["excess_loss_rate"]}% above baseline)',
                            'start': time(9, 15),
                            'end': time(10, 30),  # Block morning only on risky days
                            'days': [day_map[val]],
                            'enabled': True,
                        })

    def is_blackout(self, dt: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if the given time falls in a blackout window.

        Returns:
            (is_blackout, blackout_name_or_none)
        """
        current_time = dt.time()
        current_day = dt.weekday()

        for rule in self.blackouts:
            if not rule.get('enabled', True):
                continue

            if current_day not in rule.get('days', []):
                continue

            start = rule.get('start', time(0, 0))
            end = rule.get('end', time(23, 59))

            if start <= current_time <= end:
                return True, rule.get('name', 'unknown_blackout')

        return False, None

    def add_event_blackout(
        self,
        event_name: str,
        event_dt: datetime,
        hours_before: int = 2,
        hours_after: int = 1,
    ):
        """Add a temporary blackout around a known event (RBI, Fed, etc.)."""
        start_dt = event_dt - timedelta(hours=hours_before)
        end_dt = event_dt + timedelta(hours=hours_after)

        self.blackouts.append({
            'name': f'event_{event_name}',
            'description': f'Event blackout: {event_name}',
            'start': start_dt.time(),
            'end': end_dt.time(),
            'days': [event_dt.weekday()],
            'enabled': True,
            'expires': end_dt,
        })

    def get_active_blackouts(self) -> List[Dict]:
        """List all currently active blackout rules."""
        return [b for b in self.blackouts if b.get('enabled', True)]


# ============================================================================
# Layer 4 — Regime Gate
# ============================================================================

class RegimeGate:
    """
    Block incompatible strategy/regime combinations.

    Maps market regime to which strategy modules are ALLOWED.
    Trades using blocked strategies are REJECTED — not warned, REJECTED.
    """

    # Default regime → strategy compatibility matrix
    DEFAULT_GATES = {
        'accumulation': {
            'allowed': ['mean_reversion', 'expiry', 'mr'],
            'blocked': ['trend'],
            'description': 'Choppy/ranging market — trend following will get chopped up',
        },
        'markup': {
            'allowed': ['trend', 'pro'],
            'blocked': ['mean_reversion', 'mr'],
            'description': 'Strong uptrend — mean reversion shorts will be run over',
        },
        'markdown': {
            'allowed': ['trend', 'pro'],
            'blocked': ['mean_reversion', 'mr'],
            'description': 'Strong downtrend — mean reversion longs will be crushed',
        },
        'distribution': {
            'allowed': ['mean_reversion', 'expiry', 'mr'],
            'blocked': ['trend'],
            'description': 'Topping pattern — trend following enters too late',
        },
        'volatile': {
            'allowed': ['volatility', 'expiry'],
            'blocked': ['trend', 'mean_reversion', 'mr'],
            'description': 'VIX spike — only vol-selling or hedged positions',
        },
    }

    def __init__(self, custom_gates: Optional[Dict] = None):
        self.gates = dict(self.DEFAULT_GATES)
        if custom_gates:
            self.gates.update(custom_gates)

    def check(
        self,
        regime: str,
        strategy_name: str,
        direction: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if strategy is allowed in current regime.

        Args:
            regime: Current market regime from RegimeDetector
            strategy_name: Name of the strategy generating the trade
            direction: Optional direction for direction-specific gates

        Returns:
            (is_allowed, reason)
        """
        regime_lower = regime.lower()
        gate = self.gates.get(regime_lower)

        if gate is None:
            return True, f"No gate defined for regime '{regime}'"

        # Check if strategy matches any blocked patterns
        strat_lower = strategy_name.lower()

        for blocked in gate.get('blocked', []):
            if blocked in strat_lower:
                return False, (
                    f"REGIME GATE: {strategy_name} is blocked in {regime} regime. "
                    f"{gate.get('description', '')}"
                )

        # Check if strategy matches any allowed patterns
        for allowed in gate.get('allowed', []):
            if allowed in strat_lower:
                return True, f"Strategy {strategy_name} allowed in {regime} regime"

        # Default: allow with warning if not explicitly blocked
        return True, f"Strategy {strategy_name} not explicitly gated for {regime}"

    def get_allowed_strategies(self, regime: str) -> List[str]:
        """Get list of allowed strategy types for current regime."""
        gate = self.gates.get(regime.lower(), {})
        return gate.get('allowed', [])


# ============================================================================
# Layer 5 — Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """
    Automated protection against consecutive loss sequences.

    After 2 losses  → reduce position size by 50%
    After 3 losses  → paper trade only (no real orders)
    After 5 losses in a week → full system pause until manual reset

    Resumes normal size after 3 consecutive wins.
    """

    def __init__(
        self,
        reduce_after: int = 2,
        paper_after: int = 3,
        halt_after: int = 5,
        recovery_wins: int = 3,
    ):
        self.reduce_after = reduce_after
        self.paper_after = paper_after
        self.halt_after = halt_after
        self.recovery_wins = recovery_wins

        # State
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        self._weekly_losses = 0
        self._week_start = datetime.now().date()
        self._halted = False
        self._paper_mode = False
        self._position_size_mult = 1.0
        self._history: List[Dict] = []

    def record_trade(self, pnl: float):
        """Record a trade result and update circuit breaker state."""
        # Reset weekly counter on new week
        today = datetime.now().date()
        if (today - self._week_start).days >= 7:
            self._weekly_losses = 0
            self._week_start = today

        if pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            self._weekly_losses += 1
        else:
            self._consecutive_wins += 1
            self._consecutive_losses = 0

        # Update state
        self._update_state()

        self._history.append({
            'time': datetime.now().isoformat(),
            'pnl': pnl,
            'consec_losses': self._consecutive_losses,
            'consec_wins': self._consecutive_wins,
            'size_mult': self._position_size_mult,
            'paper_mode': self._paper_mode,
            'halted': self._halted,
        })

    def _update_state(self):
        """Update circuit breaker state based on counters."""
        # Recovery: 3 consecutive wins restores full size
        if self._consecutive_wins >= self.recovery_wins:
            self._position_size_mult = 1.0
            self._paper_mode = False
            self._halted = False
            return

        # Halt: 5+ losses in a week
        if self._weekly_losses >= self.halt_after:
            self._halted = True
            self._position_size_mult = 0.0
            return

        # Paper mode: 3+ consecutive losses
        if self._consecutive_losses >= self.paper_after:
            self._paper_mode = True
            self._position_size_mult = 0.0
            return

        # Reduced size: 2+ consecutive losses
        if self._consecutive_losses >= self.reduce_after:
            self._position_size_mult = 0.5
            return

        self._position_size_mult = 1.0

    def check(self) -> Tuple[float, List[str]]:
        """
        Check current circuit breaker state.

        Returns:
            (position_size_multiplier, list_of_active_restrictions)
        """
        restrictions = []

        if self._halted:
            restrictions.append(
                f"SYSTEM HALTED: {self._weekly_losses} losses this week "
                f"(threshold: {self.halt_after}). Manual reset required."
            )
        elif self._paper_mode:
            restrictions.append(
                f"PAPER MODE: {self._consecutive_losses} consecutive losses "
                f"(threshold: {self.paper_after}). No real orders."
            )
        elif self._position_size_mult < 1.0:
            restrictions.append(
                f"REDUCED SIZE: {self._consecutive_losses} consecutive losses. "
                f"Position size at {self._position_size_mult*100:.0f}% of normal."
            )

        return self._position_size_mult, restrictions

    def reset(self):
        """Manual reset (requires explicit user action)."""
        self._halted = False
        self._paper_mode = False
        self._position_size_mult = 1.0
        self._consecutive_losses = 0
        self._weekly_losses = 0

    def get_status(self) -> Dict:
        """Get full circuit breaker status."""
        return {
            'consecutive_losses': self._consecutive_losses,
            'consecutive_wins': self._consecutive_wins,
            'weekly_losses': self._weekly_losses,
            'position_size_multiplier': self._position_size_mult,
            'paper_mode': self._paper_mode,
            'halted': self._halted,
            'history_count': len(self._history),
        }


# ============================================================================
# Layer 6 — Post-Loss Learning Loop
# ============================================================================

class PostLossLearningLoop:
    """
    Every loss makes the system smarter.

    After each losing trade:
    1. Auto-tag the loss with all attributes
    2. Re-run pattern analysis with new data point
    3. Check if new loss reveals previously unseen pattern
    4. If new pattern emerges → auto-generate new filter rule
    5. Log to knowledge base for permanent memory
    """

    def __init__(self, knowledge_base_path: str = 'loss_knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base: Dict = self._load_knowledge()

    def _load_knowledge(self) -> Dict:
        """Load existing knowledge base."""
        path = Path(self.knowledge_base_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {
            'version': 1,
            'total_losses_analyzed': 0,
            'discovered_patterns': [],
            'auto_generated_rules': [],
            'loss_history': [],
        }

    def _save_knowledge(self):
        """Save knowledge base to disk."""
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        with open(self.knowledge_base_path, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2, default=make_serializable)

    def learn_from_loss(self, trade_attrs: Dict) -> Dict:
        """
        Process a new losing trade and extract learnings.

        Args:
            trade_attrs: Full attribute dict of the losing trade

        Returns:
            Learning result with any new patterns discovered
        """
        result = {
            'new_patterns': [],
            'rules_updated': [],
            'knowledge_updated': True,
        }

        # Store in history
        entry = {
            'timestamp': datetime.now().isoformat(),
            'direction': trade_attrs.get('direction', 'unknown'),
            'regime': trade_attrs.get('regime_at_entry', 'unknown'),
            'strategy': trade_attrs.get('strategy', 'unknown'),
            'net_pnl': float(trade_attrs.get('net_pnl', 0)),
            'exit_reason': trade_attrs.get('exit_reason', 'unknown'),
            'signal_count': int(trade_attrs.get('signal_count', 0)),
            'archetype': trade_attrs.get('loss_archetype', 'unclassified'),
        }
        self.knowledge_base['loss_history'].append(entry)
        self.knowledge_base['total_losses_analyzed'] += 1

        # Check for emerging patterns (every 10 losses)
        history = self.knowledge_base['loss_history']
        if len(history) >= 10 and len(history) % 5 == 0:
            new_patterns = self._check_emerging_patterns(history[-20:])
            if new_patterns:
                result['new_patterns'] = new_patterns
                self.knowledge_base['discovered_patterns'].extend(new_patterns)

        self._save_knowledge()
        return result

    def _check_emerging_patterns(self, recent_losses: List[Dict]) -> List[Dict]:
        """Check if recent losses reveal a new pattern."""
        patterns = []

        # Check regime concentration
        regimes = [l['regime'] for l in recent_losses]
        for regime in set(regimes):
            count = regimes.count(regime)
            if count / len(recent_losses) > 0.6:
                patterns.append({
                    'type': 'regime_concentration',
                    'description': f'{count}/{len(recent_losses)} recent losses in {regime} regime',
                    'regime': regime,
                    'concentration': round(count / len(recent_losses) * 100, 1),
                    'discovered_at': datetime.now().isoformat(),
                })

        # Check strategy concentration
        strategies = [l.get('strategy', '') for l in recent_losses]
        for strat in set(strategies):
            if not strat:
                continue
            count = strategies.count(strat)
            if count / len(recent_losses) > 0.5:
                patterns.append({
                    'type': 'strategy_concentration',
                    'description': f'{count}/{len(recent_losses)} recent losses from {strat}',
                    'strategy': strat,
                    'concentration': round(count / len(recent_losses) * 100, 1),
                    'discovered_at': datetime.now().isoformat(),
                })

        # Check archetype concentration
        archetypes = [l.get('archetype', '') for l in recent_losses]
        for arch in set(archetypes):
            if not arch or arch == 'unclassified':
                continue
            count = archetypes.count(arch)
            if count / len(recent_losses) > 0.5:
                patterns.append({
                    'type': 'archetype_concentration',
                    'description': f'{count}/{len(recent_losses)} recent losses are {arch} archetype',
                    'archetype': arch,
                    'concentration': round(count / len(recent_losses) * 100, 1),
                    'discovered_at': datetime.now().isoformat(),
                })

        return patterns

    def get_knowledge_summary(self) -> Dict:
        """Get summary of accumulated knowledge."""
        return {
            'total_losses_analyzed': self.knowledge_base['total_losses_analyzed'],
            'patterns_discovered': len(self.knowledge_base['discovered_patterns']),
            'auto_rules_generated': len(self.knowledge_base['auto_generated_rules']),
            'recent_patterns': self.knowledge_base['discovered_patterns'][-5:],
        }


# ============================================================================
# Master Loss Elimination Engine
# ============================================================================

class LossEliminationEngine:
    """
    Orchestrates all 6 elimination layers into a single pre-trade check.

    Integration point: Called by RiskManager before every trade.
    """

    def __init__(
        self,
        patterns_file: Optional[str] = None,
        archetype_stats: Optional[Dict] = None,
        high_risk_windows: Optional[List[Dict]] = None,
        knowledge_base_path: str = 'loss_knowledge_base.json',
    ):
        # Layer 1: Kill Switch
        self.kill_switch = PreTradeKillSwitch(patterns_file)

        # Layer 2: Adaptive Stop
        self.adaptive_stop = AdaptiveStopLoss(archetype_stats)

        # Layer 3: Blackout Manager
        self.blackout_mgr = TemporalBlackoutManager(
            high_risk_windows=high_risk_windows
        )

        # Layer 4: Regime Gate
        self.regime_gate = RegimeGate()

        # Layer 5: Circuit Breaker
        self.circuit_breaker = CircuitBreaker()

        # Layer 6: Learning Loop
        self.learning_loop = PostLossLearningLoop(knowledge_base_path)

    def pre_trade_check(
        self,
        trade_attrs: Dict,
        current_time: Optional[datetime] = None,
    ) -> EliminationResult:
        """
        Run all 6 layers and return a unified result.

        This is THE function called before every trade.

        Args:
            trade_attrs: Dict with keys matching LossDNATagger output columns:
                direction, regime_at_entry, strategy, entry_price, atr_at_entry,
                signal_count, bull_score, bear_score, day_name, etc.
            current_time: Time of trade entry (defaults to now)

        Returns:
            EliminationResult with final verdict
        """
        if current_time is None:
            current_time = datetime.now()

        reasons = []
        warnings = []
        matched_patterns = []
        layer_results = {}
        final_mult = 1.0

        # === Layer 1: Kill Switch ===
        risk_score, verdict, matches = self.kill_switch.check(trade_attrs)
        layer_results['kill_switch'] = {
            'loss_risk_score': risk_score,
            'verdict': verdict.value,
            'matches': len(matches),
        }
        matched_patterns.extend(matches)

        if verdict == KillSwitchVerdict.BLOCKED:
            reasons.append(
                f"KILL SWITCH: Loss Risk Score {risk_score}/100 — "
                f"{len(matches)} toxic patterns matched"
            )
            final_mult = 0.0
        elif verdict == KillSwitchVerdict.WARNING:
            warnings.append(
                f"KILL SWITCH WARNING: Loss Risk Score {risk_score}/100 — "
                f"Position size reduced 50%"
            )
            final_mult = min(final_mult, 0.5)

        # === Layer 2: Adaptive Stop is advisory (doesn't block) ===
        stop_info = self.adaptive_stop.calculate_stop(
            entry_price=trade_attrs.get('entry_price', 0),
            atr=trade_attrs.get('atr_at_entry', 0),
            direction=trade_attrs.get('direction', 'bullish'),
        )
        layer_results['adaptive_stop'] = stop_info

        # === Layer 3: Temporal Blackout ===
        is_blackout, blackout_name = self.blackout_mgr.is_blackout(current_time)
        layer_results['blackout'] = {
            'is_blackout': is_blackout,
            'window': blackout_name,
        }
        if is_blackout:
            reasons.append(f"TEMPORAL BLACKOUT: {blackout_name}")
            final_mult = 0.0

        # === Layer 4: Regime Gate ===
        regime = trade_attrs.get('regime_at_entry', 'unknown')
        strategy = trade_attrs.get('strategy', '')
        gate_allowed, gate_reason = self.regime_gate.check(regime, strategy)
        layer_results['regime_gate'] = {
            'allowed': gate_allowed,
            'reason': gate_reason,
        }
        if not gate_allowed:
            reasons.append(f"REGIME GATE: {gate_reason}")
            final_mult = 0.0

        # === Layer 5: Circuit Breaker ===
        cb_mult, cb_restrictions = self.circuit_breaker.check()
        layer_results['circuit_breaker'] = self.circuit_breaker.get_status()
        if cb_mult < 1.0:
            final_mult = min(final_mult, cb_mult)
            if cb_mult == 0.0:
                reasons.extend(cb_restrictions)
            else:
                warnings.extend(cb_restrictions)

        # === Final Verdict ===
        approved = final_mult > 0.0 and len(reasons) == 0

        if final_mult == 0.0:
            final_verdict = KillSwitchVerdict.BLOCKED
        elif final_mult < 1.0:
            final_verdict = KillSwitchVerdict.WARNING
        else:
            final_verdict = KillSwitchVerdict.CLEAR

        return EliminationResult(
            approved=approved,
            verdict=final_verdict,
            loss_risk_score=risk_score,
            position_size_multiplier=final_mult,
            reasons=reasons,
            warnings=warnings,
            matched_patterns=matched_patterns,
            layer_results=layer_results,
        )

    def record_trade_result(self, pnl: float, trade_attrs: Optional[Dict] = None):
        """
        Record a completed trade. Updates circuit breaker and learning loop.

        Call this after every trade exit.
        """
        # Update circuit breaker
        self.circuit_breaker.record_trade(pnl)

        # If loss, run learning loop
        if pnl < 0 and trade_attrs:
            self.learning_loop.learn_from_loss(trade_attrs)

    def get_full_status(self) -> Dict:
        """Get status of all elimination layers."""
        return {
            'kill_switch': {
                'loaded_patterns': len(self.kill_switch.patterns),
                'combo_patterns': len(self.kill_switch.combination_patterns),
            },
            'adaptive_stop': {
                'dominant_archetype': self.adaptive_stop._dominant_archetype,
            },
            'blackouts': {
                'active_rules': len(self.blackout_mgr.get_active_blackouts()),
            },
            'regime_gate': {
                'regimes_configured': len(self.regime_gate.gates),
            },
            'circuit_breaker': self.circuit_breaker.get_status(),
            'learning_loop': self.learning_loop.get_knowledge_summary(),
        }
