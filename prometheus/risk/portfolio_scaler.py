import pandas as pd
import numpy as np

class RiskPortfolioScaler:
    """
    Workstream E: Risk Management and Post-Tax Sizing Framework
    """
    def __init__(self, initial_equity):
        self.peak_equity = initial_equity
        self.current_equity = initial_equity

    def update_equity(self, active_equity):
        self.current_equity = active_equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def get_drawdown_multiplier(self):
        """
        Component E1: Drawdown-adaptive sizing block.
        Non-negotiable capital preservation via mechanical exposure suppression.
        """
        if self.peak_equity <= 0:
            return 1.0
            
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        
        if dd > 0.10:
            return 0.0 # Halt trading fully (10% DD rule)
        elif dd > 0.05:
            return 0.5 # Halve position sizes (5% DD rule)
        return 1.0

    def get_correlation_multiplier(self, returns_strategy_a, returns_strategy_b, window=20):
        """
        Component E2: Cross-strategy correlation tracker.
        Halves size across the board if VRP + Window Classifier PnL highly correlates.
        Expected inputs: Arrays or pandas Series of daily PnL relative returns.
        """
        if len(returns_strategy_a) < window or len(returns_strategy_b) < window:
            return 1.0
            
        s1 = pd.Series(returns_strategy_a[-window:])
        s2 = pd.Series(returns_strategy_b[-window:])
        corr = s1.corr(s2)
        
        if pd.isna(corr):
            return 1.0
            
        if corr > 0.60:
            return 0.5 # System acts as single directional hazard. Cut size.
        return 1.0

def calculate_net_pnl_india(gross_pnl, product_type, entry_price, exit_price, qty):
    """
    Component E3: Tax and cost accounting (Indian F&O Markets Context).
    Removes edge illusions via STT, GST, and Stamp duties.
    """
    turnover = (entry_price + exit_price) * qty
    brokerage_flat = 40.0 # Combined buy + sell standard flat rate.
    
    stt = 0.0
    exchange_txn = 0.0
    # Assuming standard NIFTY indices multiplier parameters
    if product_type == "FUTURES":
        # STT applied on sell side turnover
        stt = exit_price * qty * 0.0125 / 100 
        exchange_txn = turnover * 0.00002
    elif product_type == "OPTIONS":
        # STT on premium sell-side. Buy-to-close incurs no STT. Let's assume average.
        sold_premium = exit_price if gross_pnl < 0 else entry_price 
        stt = sold_premium * qty * 0.0625 / 100
        exchange_txn = turnover * 0.0005
        
    gst = (brokerage_flat + exchange_txn) * 0.18
    sebi_charges = turnover * 0.000001
    stamp_duty = turnover * 0.00002 # Buying side normally.
    
    total_friction = stt + brokerage_flat + exchange_txn + gst + sebi_charges + stamp_duty
    net_pnl = gross_pnl - total_friction
    
    return net_pnl