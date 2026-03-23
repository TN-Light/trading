# ============================================================================
# PROMETHEUS — Loss Analysis Dashboard
# ============================================================================
"""
Interactive loss analysis dashboard using Plotly.

Generates comprehensive HTML dashboard with:
  1. Loss Pattern Severity Table (top 10 patterns)
  2. Temporal Loss Heatmap (day × time × regime)
  3. Loss Archetype Pie Chart
  4. Kill Switch Risk Score Distribution
  5. Circuit Breaker Timeline
  6. Equity Curve with Drawdown Overlay
  7. Pattern Discovery Timeline
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class LossDashboard:
    """
    Generate interactive HTML dashboards from pattern mining results.
    """

    # Color palette (dark premium theme)
    COLORS = {
        'bg': '#0d1117',
        'card_bg': '#161b22',
        'text': '#c9d1d9',
        'accent': '#58a6ff',
        'red': '#f85149',
        'green': '#3fb950',
        'yellow': '#d29922',
        'orange': '#db6d28',
        'purple': '#bc8cff',
        'cyan': '#39d353',
        'muted': '#8b949e',
    }

    ARCHETYPE_COLORS = {
        'false_signal': '#f85149',
        'stop_hunt': '#db6d28',
        'chop_grind': '#d29922',
        'overextension': '#bc8cff',
        'oversizing': '#ff7b72',
        'sequence': '#58a6ff',
        'cost_kill': '#8b949e',
        'regime_mismatch': '#79c0ff',
        'unclassified': '#484f58',
    }

    def __init__(self, title: str = "PROMETHEUS Loss DNA Dashboard"):
        self.title = title
        self.figures = []

    def generate(
        self,
        tagged_df: pd.DataFrame,
        analysis_results: Dict,
        output_path: str = 'loss_dashboard.html',
    ):
        """
        Generate full dashboard.

        Args:
            tagged_df: Output from LossDNATagger.tag_trades() with archetypes
            analysis_results: Output from PatternMiningEngine.run_full_analysis()
            output_path: Where to save the HTML
        """
        if not HAS_PLOTLY:
            print("  ✗ Plotly not installed. Run: pip install plotly")
            return

        figs = []

        # 1. Summary KPI Cards (as table)
        figs.append(self._create_summary_kpis(tagged_df, analysis_results))

        # 2. Equity Curve with Drawdown
        figs.append(self._create_equity_curve(tagged_df))

        # 3. Loss Archetype Breakdown
        archetype_data = analysis_results.get('layers', {}).get('archetypes', {})
        if archetype_data:
            figs.append(self._create_archetype_chart(archetype_data))

        # 4. Top Loss Patterns Table
        freq_data = analysis_results.get('layers', {}).get('frequency', {})
        if freq_data.get('patterns'):
            figs.append(self._create_pattern_table(freq_data['patterns'][:15]))

        # 5. Temporal Heatmap (day × regime)
        temporal = analysis_results.get('layers', {}).get('temporal', {})
        if temporal.get('heatmaps'):
            figs.append(self._create_temporal_heatmap(tagged_df))

        # 6. Combination Patterns Visualization
        combo_data = analysis_results.get('layers', {}).get('combination', {})
        if combo_data.get('patterns'):
            figs.append(self._create_combo_chart(combo_data['patterns'][:15]))

        # 7. ML Feature Importance
        ml_data = analysis_results.get('layers', {}).get('hidden_ml', {})
        if ml_data.get('feature_importance_top20'):
            figs.append(self._create_feature_importance(ml_data))

        # 8. Loss Severity Distribution
        figs.append(self._create_loss_distribution(tagged_df))

        # Build HTML
        self._save_dashboard(figs, output_path)
        print(f"\n  ✓ Dashboard saved: {output_path}")

    def _create_summary_kpis(self, df: pd.DataFrame, results: Dict) -> Any:
        """Create summary KPI display."""
        summary = results.get('summary', {})
        freq = results.get('layers', {}).get('frequency', {}).get('summary', {})
        combo = results.get('layers', {}).get('combination', {})

        losers = df[~df['win']]
        total_loss_pnl = losers['net_pnl'].sum()
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color=self.COLORS['card_bg'],
                font=dict(color=self.COLORS['accent'], size=14),
                align='left',
            ),
            cells=dict(
                values=[
                    ['Total Trades', 'Win Rate', 'Total Losses', 'Total Loss P&L',
                     'Avg Loss', 'Loss Patterns Found', 'Toxic Combos Found',
                     'Lethal Patterns', 'Critical Patterns'],
                    [summary.get('total_trades', 0),
                     f"{summary.get('win_rate', 0)}%",
                     summary.get('losses', 0),
                     f"₹{total_loss_pnl:,.0f}",
                     f"₹{avg_loss:,.0f}",
                     freq.get('total_patterns', 0),
                     combo.get('total_found', 0),
                     freq.get('lethal_patterns', 0),
                     freq.get('critical_patterns', 0)],
                ],
                fill_color=self.COLORS['bg'],
                font=dict(color=self.COLORS['text'], size=13),
                align='left',
                height=30,
            )
        ))
        fig.update_layout(
            title='Loss DNA Summary',
            paper_bgcolor=self.COLORS['bg'],
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    def _create_equity_curve(self, df: pd.DataFrame) -> Any:
        """Equity curve with drawdown overlay."""
        df_sorted = df.sort_values('entry_dt')
        cum_pnl = df_sorted['net_pnl'].cumsum()
        peak = cum_pnl.cummax()
        drawdown = cum_pnl - peak

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )

        # Equity curve
        colors = [self.COLORS['green'] if p > 0 else self.COLORS['red']
                  for p in df_sorted['net_pnl']]

        fig.add_trace(go.Scatter(
            x=list(range(len(cum_pnl))),
            y=cum_pnl.values,
            mode='lines',
            line=dict(color=self.COLORS['accent'], width=2),
            name='Equity',
        ), row=1, col=1)

        # Mark losses
        loss_indices = df_sorted[~df_sorted['win']].index
        loss_positions = [list(df_sorted.index).index(i) for i in loss_indices if i in df_sorted.index]
        if loss_positions:
            fig.add_trace(go.Scatter(
                x=loss_positions,
                y=cum_pnl.iloc[loss_positions].values,
                mode='markers',
                marker=dict(color=self.COLORS['red'], size=5, symbol='x'),
                name='Losses',
            ), row=1, col=1)

        # Drawdown
        fig.add_trace(go.Scatter(
            x=list(range(len(drawdown))),
            y=drawdown.values,
            fill='tonexty',
            line=dict(color=self.COLORS['red'], width=1),
            fillcolor='rgba(248, 81, 73, 0.15)',
            name='Drawdown',
        ), row=2, col=1)

        fig.update_layout(
            title='Equity Curve with Loss Markers & Drawdown',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['card_bg'],
            font=dict(color=self.COLORS['text']),
            height=500,
            showlegend=True,
            legend=dict(bgcolor=self.COLORS['card_bg']),
        )
        fig.update_xaxes(gridcolor='#21262d')
        fig.update_yaxes(gridcolor='#21262d')

        return fig

    def _create_archetype_chart(self, archetype_data: Dict) -> Any:
        """Pie chart of loss archetypes."""
        labels = []
        values = []
        colors = []
        texts = []

        for arch, data in archetype_data.items():
            labels.append(data.get('name', arch))
            values.append(data['count'])
            colors.append(self.ARCHETYPE_COLORS.get(arch, '#484f58'))
            texts.append(f"₹{data['total_pnl']:,.0f} | Fix: {data.get('fix', 'N/A')}")

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textfont=dict(size=12),
            hovertext=texts,
            hole=0.4,
        )])

        fig.update_layout(
            title='Loss Archetype Breakdown',
            paper_bgcolor=self.COLORS['bg'],
            font=dict(color=self.COLORS['text']),
            height=450,
            legend=dict(bgcolor=self.COLORS['card_bg']),
        )
        return fig

    def _create_pattern_table(self, patterns: list) -> Any:
        """Top loss patterns table."""
        attrs = [p['attribute'] for p in patterns]
        vals = [p['value'] for p in patterns]
        scores = [p['loss_frequency_score'] for p in patterns]
        severities = [p['severity'] for p in patterns]
        loss_pcts = [f"{p['loss_pct']}%" for p in patterns]
        win_pcts = [f"{p['win_pct']}%" for p in patterns]

        severity_colors = {
            'lethal': self.COLORS['red'],
            'critical': self.COLORS['orange'],
            'loss_pattern': self.COLORS['yellow'],
            'normal': self.COLORS['muted'],
        }

        cell_colors = [
            [self.COLORS['bg']] * len(patterns),
            [self.COLORS['bg']] * len(patterns),
            [self.COLORS['bg']] * len(patterns),
            [severity_colors.get(s, self.COLORS['bg']) for s in severities],
            [self.COLORS['bg']] * len(patterns),
            [self.COLORS['bg']] * len(patterns),
        ]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Attribute', 'Value', 'Score', 'Severity', 'Loss%', 'Win%'],
                fill_color=self.COLORS['card_bg'],
                font=dict(color=self.COLORS['accent'], size=13),
                align='left',
            ),
            cells=dict(
                values=[attrs, vals, scores, severities, loss_pcts, win_pcts],
                fill_color=cell_colors,
                font=dict(color=self.COLORS['text'], size=12),
                align='left',
                height=28,
            )
        )])

        fig.update_layout(
            title='Top Loss Patterns (Frequency Mining)',
            paper_bgcolor=self.COLORS['bg'],
            height=max(300, 50 + len(patterns) * 30),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    def _create_temporal_heatmap(self, df: pd.DataFrame) -> Any:
        """Day of week × regime loss rate heatmap."""
        if 'day_name' not in df.columns or 'regime_at_entry' not in df.columns:
            return go.Figure()

        pivot = df.pivot_table(
            values='win',
            index='regime_at_entry',
            columns='day_name',
            aggfunc=lambda x: round((1 - x.mean()) * 100, 1),
        )

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        available_days = [d for d in day_order if d in pivot.columns]
        pivot = pivot.reindex(columns=available_days)

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0, self.COLORS['green']],
                [0.5, self.COLORS['yellow']],
                [1, self.COLORS['red']],
            ],
            text=pivot.values,
            texttemplate='%{text}%',
            textfont=dict(size=12),
            colorbar=dict(
                title='Loss Rate %',
                tickfont=dict(color=self.COLORS['text']),
                titlefont=dict(color=self.COLORS['text']),
            ),
        ))

        fig.update_layout(
            title='Loss Rate Heatmap: Regime × Day of Week',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['card_bg'],
            font=dict(color=self.COLORS['text']),
            height=400,
        )
        return fig

    def _create_combo_chart(self, combos: list) -> Any:
        """Top toxic combination patterns bar chart."""
        labels = [' + '.join(c['combination']) for c in combos]
        loss_rates = [c['loss_rate'] for c in combos]
        counts = [c['count'] for c in combos]

        # Truncate labels for readability
        labels = [l[:60] + '...' if len(l) > 60 else l for l in labels]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=loss_rates[::-1],
            orientation='h',
            marker=dict(
                color=loss_rates[::-1],
                colorscale=[[0, self.COLORS['yellow']], [1, self.COLORS['red']]],
            ),
            text=[f"{r}% ({c} trades)" for r, c in zip(loss_rates[::-1], counts[::-1])],
            textposition='outside',
            textfont=dict(color=self.COLORS['text'], size=11),
        ))

        fig.update_layout(
            title='Top Toxic Combinations (Apriori Mining)',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['card_bg'],
            font=dict(color=self.COLORS['text'], size=10),
            height=max(400, len(combos) * 35),
            xaxis_title='Loss Rate %',
            margin=dict(l=300, r=120, t=60, b=40),
        )
        fig.update_xaxes(gridcolor='#21262d')
        fig.update_yaxes(gridcolor='#21262d')

        return fig

    def _create_feature_importance(self, ml_data: Dict) -> Any:
        """ML feature importance bar chart."""
        importance = ml_data.get('feature_importance_top20', {})
        if not importance:
            return go.Figure()

        features = list(importance.keys())[:15]
        values = [importance[f] for f in features]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features[::-1],
            x=values[::-1],
            orientation='h',
            marker=dict(color=self.COLORS['purple']),
        ))

        accuracy = ml_data.get('cv_accuracy', 0)
        fig.update_layout(
            title=f'XGBoost Feature Importance (CV Accuracy: {accuracy}%)',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['card_bg'],
            font=dict(color=self.COLORS['text']),
            height=max(400, len(features) * 30),
            xaxis_title='Importance',
            margin=dict(l=250, r=40, t=60, b=40),
        )
        fig.update_xaxes(gridcolor='#21262d')
        fig.update_yaxes(gridcolor='#21262d')

        return fig

    def _create_loss_distribution(self, df: pd.DataFrame) -> Any:
        """Loss P&L distribution histogram."""
        losers = df[~df['win']]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=losers['net_pnl'],
            nbinsx=30,
            marker=dict(
                color=self.COLORS['red'],
                line=dict(color=self.COLORS['card_bg'], width=1),
            ),
            name='Losses',
        ))

        # Add severity boundaries
        for val, label, color in [
            (-100, 'Small/Medium', self.COLORS['yellow']),
            (-200, 'Medium/Large', self.COLORS['orange']),
            (-400, 'Large/Catastrophic', self.COLORS['red']),
        ]:
            fig.add_vline(x=val, line_dash='dash', line_color=color,
                         annotation_text=label,
                         annotation_font_color=color)

        fig.update_layout(
            title='Loss P&L Distribution by Severity',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['card_bg'],
            font=dict(color=self.COLORS['text']),
            height=400,
            xaxis_title='P&L (₹)',
            yaxis_title='Count',
        )
        fig.update_xaxes(gridcolor='#21262d')
        fig.update_yaxes(gridcolor='#21262d')

        return fig

    def _save_dashboard(self, figures: list, output_path: str):
        """Combine all figures into a single HTML dashboard."""
        html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        body {{
            background: {self.COLORS['bg']};
            color: {self.COLORS['text']};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .dashboard-header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #21262d;
            margin-bottom: 30px;
        }}
        .dashboard-header h1 {{
            color: {self.COLORS['accent']};
            font-size: 28px;
            margin: 0 0 10px 0;
        }}
        .dashboard-header p {{
            color: {self.COLORS['muted']};
            font-size: 14px;
        }}
        .chart-container {{
            background: {self.COLORS['card_bg']};
            border: 1px solid #21262d;
            border-radius: 8px;
            margin: 20px 0;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🔬 {self.title}</h1>
        <p>Systematic Loss Pattern Discovery & Elimination Report</p>
    </div>
"""]

        for fig in figures:
            html_parts.append('<div class="chart-container">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            html_parts.append('</div>')

        html_parts.append('</body></html>')

        Path(output_path).write_text('\n'.join(html_parts), encoding='utf-8')
