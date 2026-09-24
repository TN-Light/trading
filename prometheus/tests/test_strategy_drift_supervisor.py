import pytest
from prometheus.risk.strategy_drift_supervisor import StrategyDriftSupervisor
from prometheus.risk.portfolio_scaler import RiskPortfolioScaler
from prometheus.risk.manager import RiskManager


def test_drift_supervisor_insufficient_data():
    supervisor = StrategyDriftSupervisor(min_trades_for_eval=5)
    # Only 3 trades
    supervisor.record_trade(pnl=100.0)
    supervisor.record_trade(pnl=-50.0)
    supervisor.record_trade(pnl=200.0)

    health = supervisor.evaluate_regime_health()
    assert health["status"] == "INSUFFICIENT_DATA"
    assert health["multiplier"] == 1.0
    assert supervisor.get_drift_multiplier() == 1.0


def test_drift_supervisor_healthy_regime():
    supervisor = StrategyDriftSupervisor(min_trades_for_eval=5, healthy_pf_threshold=1.30)
    # 5 trades with 4 wins, 1 loss -> Gross Win = 600, Gross Loss = 100 -> PF = 6.0
    supervisor.record_trade(pnl=150.0)
    supervisor.record_trade(pnl=200.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=120.0)
    supervisor.record_trade(pnl=130.0)

    health = supervisor.evaluate_regime_health()
    assert health["status"] == "HEALTHY"
    assert health["multiplier"] == 1.0
    assert health["profit_factor"] == 6.0
    assert health["win_rate"] == 80.0
    assert supervisor.get_drift_multiplier() == 1.0


def test_drift_supervisor_degraded_regime():
    supervisor = StrategyDriftSupervisor(min_trades_for_eval=5, healthy_pf_threshold=1.30, degraded_pf_threshold=1.00)
    # Gross Win = 220, Gross Loss = 200 -> PF = 1.10 (between 1.0 and 1.3)
    supervisor.record_trade(pnl=110.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=110.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=0.0)

    health = supervisor.evaluate_regime_health()
    assert health["status"] == "DEGRADED"
    assert health["multiplier"] == 0.5
    assert health["profit_factor"] == 1.1
    assert supervisor.get_drift_multiplier() == 0.5


def test_drift_supervisor_decaying_quarantine():
    supervisor = StrategyDriftSupervisor(min_trades_for_eval=5, degraded_pf_threshold=1.00)
    # Severe decay: Gross Win = 50, Gross Loss = 400 -> PF = 0.125 (< 1.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=50.0)
    supervisor.record_trade(pnl=-100.0)
    supervisor.record_trade(pnl=-100.0)

    health = supervisor.evaluate_regime_health()
    assert health["status"] == "DECAYING"
    assert health["multiplier"] == 0.0
    assert supervisor.get_drift_multiplier() == 0.0


def test_risk_manager_drift_integration():
    rm = RiskManager(config={}, initial_capital=100000.0)
    # Force drift supervisor on rm to simulated decay
    for _ in range(5):
        rm.record_trade_result(pnl=-500.0, trade={"strategy": "Momentum"})

    # Position sizing should now be quarantined (0 lots)
    sizing = rm.calculate_position_size(
        entry_price=100.0,
        stop_loss=90.0,
        lot_size=65,
        risk_per_trade_pct=2.0
    )
    assert sizing["lots"] == 0
    assert "quarantined" in sizing["error"].lower()

