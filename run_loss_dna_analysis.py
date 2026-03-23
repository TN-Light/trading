# ============================================================================
# PROMETHEUS — Loss DNA Analysis: Master Runner
# ============================================================================
"""
Run the complete Loss DNA Analysis pipeline on your trade data.

Usage:
    python run_loss_dna_analysis.py

This script:
  1. Loads trade CSVs
  2. Tags every trade with 40+ attributes (LossDNATagger)
  3. Runs 5-layer pattern mining (PatternMiningEngine)
  4. Classifies losses into 8 archetypes
  5. Generates interactive HTML dashboard
  6. Generates plain-language markdown report
  7. Initializes the LossEliminationEngine with discovered patterns
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from prometheus.analysis.loss_dna_tagger import LossDNATagger
from prometheus.analysis.pattern_miner import PatternMiningEngine
from prometheus.analysis.loss_dashboard import LossDashboard
from prometheus.analysis.loss_report_generator import LossReportGenerator
from prometheus.risk.loss_elimination_engine import LossEliminationEngine


def run_analysis(csv_file: str, daily_csv: str = None):
    """Run complete loss DNA analysis on a single trade CSV."""
    path = Path(csv_file)
    if not path.exists():
        print(f"  ✗ Not found: {csv_file}")
        return

    symbol = path.stem.replace('trade_analysis_', '').replace('_full', '')
    print(f"\n{'='*70}")
    print(f"  PROMETHEUS LOSS DNA ANALYSIS — {symbol}")
    print(f"{'='*70}")
    start = time.time()

    # ─── Step 1: Load Data ───────────────────────────────────────────────
    print(f"\n  → Loading trade data...")
    trades = pd.read_csv(csv_file)
    print(f"    {len(trades)} trades loaded")

    # ─── Step 2: Tag Trades ──────────────────────────────────────────────
    print(f"\n  → Step 2: Tagging trades with 40+ DNA attributes...")
    tagger = LossDNATagger()
    tagged = tagger.tag_trades(trades)
    n_attrs = len([c for c in tagged.columns if c not in trades.columns])
    print(f"    ✓ {n_attrs} new attributes added per trade")

    # Save tagged database
    db_path = f'loss_database_{symbol}.csv'
    tagger.save_loss_database(tagged, db_path)

    # ─── Step 3: Run Pattern Mining ──────────────────────────────────────
    print(f"\n  → Step 3: Running 5-layer pattern mining...")
    engine = PatternMiningEngine()
    results, tagged = engine.run_full_analysis(tagged)

    # Save pattern report
    report_path = f'loss_patterns_report_{symbol}.json'
    engine.save_report(results, report_path)

    # ─── Step 4: Generate Dashboard ──────────────────────────────────────
    print(f"\n  → Step 4: Generating interactive dashboard...")
    dashboard = LossDashboard(title=f"PROMETHEUS Loss DNA — {symbol}")
    dash_path = f'loss_dashboard_{symbol}.html'
    dashboard.generate(tagged, results, dash_path)

    # ─── Step 5: Generate Report ─────────────────────────────────────────
    print(f"\n  → Step 5: Generating loss elimination report...")
    reporter = LossReportGenerator()
    rpt_path = f'loss_elimination_report_{symbol}.md'
    reporter.generate(tagged, results, rpt_path)

    # ─── Step 6: Initialize Elimination Engine ───────────────────────────
    print(f"\n  → Step 6: Initializing Loss Elimination Engine...")
    archetypes = results.get('layers', {}).get('archetypes', {})
    temporal = results.get('layers', {}).get('temporal', {})
    high_risk = temporal.get('high_risk_windows', [])

    elimination = LossEliminationEngine(
        patterns_file=report_path,
        archetype_stats=archetypes,
        high_risk_windows=high_risk,
        knowledge_base_path=f'loss_knowledge_base_{symbol}.json',
    )
    status = elimination.get_full_status()
    print(f"    Kill Switch: {status['kill_switch']['loaded_patterns']} patterns loaded")
    print(f"    Combo Patterns: {status['kill_switch']['combo_patterns']} loaded")
    print(f"    Blackout Rules: {status['blackouts']['active_rules']} active")
    print(f"    Regime Gates: {status['regime_gate']['regimes_configured']} regimes configured")

    # ─── Step 7: Demo — Score Sample Trades ──────────────────────────────
    print(f"\n  → Step 7: Demo — scoring sample trades against elimination engine...")
    losers = tagged[~tagged['win']].head(3)
    winners = tagged[tagged['win']].head(3)

    for label, sample in [("LOSING", losers), ("WINNING", winners)]:
        for _, trade in sample.iterrows():
            attrs = trade.to_dict()
            result = elimination.pre_trade_check(attrs)
            emoji = {'clear': '✅', 'warning': '⚠️', 'blocked': '🛑'}.get(
                result.verdict.value, '❓')
            print(f"    {emoji} {label} trade ({attrs.get('strategy', '?')}, "
                  f"{attrs.get('direction', '?')}): "
                  f"Score={result.loss_risk_score}, "
                  f"Verdict={result.verdict.value}, "
                  f"Size={result.position_size_multiplier*100:.0f}%")
            if result.reasons:
                for r in result.reasons[:2]:
                    print(f"       → {r}")

    # ─── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    summary = results.get('summary', {})
    freq = results.get('layers', {}).get('frequency', {}).get('summary', {})
    combo = results.get('layers', {}).get('combination', {})
    ml = results.get('layers', {}).get('hidden_ml', {})

    print(f"\n{'─'*70}")
    print(f"  ANALYSIS COMPLETE — {symbol} ({elapsed:.1f}s)")
    print(f"{'─'*70}")
    print(f"  Trades: {summary.get('total_trades', 0)} | "
          f"Win Rate: {summary.get('win_rate', 0)}% | "
          f"P&L: ₹{summary.get('total_pnl', 0):,.0f}")
    print(f"  Patterns: {freq.get('total_patterns', 0)} found "
          f"({freq.get('lethal_patterns', 0)} lethal, "
          f"{freq.get('critical_patterns', 0)} critical)")
    print(f"  Toxic Combos: {combo.get('total_found', 0)}")
    if 'cv_accuracy' in ml:
        print(f"  ML Accuracy: {ml['cv_accuracy']}% (±{ml['cv_std']}%)")
    print(f"\n  Output Files:")
    print(f"    📊 {db_path}")
    print(f"    📋 {report_path}")
    print(f"    🎨 {dash_path}")
    print(f"    📝 {rpt_path}")


if __name__ == '__main__':
    csv_files = [
        'trade_analysis_NIFTY_50_full.csv',
        'trade_analysis_NIFTY_BANK_full.csv',
        'trade_analysis_SENSEX_full.csv',
    ]

    existing = [f for f in csv_files if Path(f).exists()]
    if not existing:
        print("  ✗ No trade CSV files found. Run backtesting first.")
        sys.exit(1)

    for csv_file in existing:
        run_analysis(csv_file)

    print(f"\n{'='*70}")
    print(f"  ALL ANALYSES COMPLETE")
    print(f"{'='*70}")
