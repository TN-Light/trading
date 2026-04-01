import datetime
import pandas as pd

class VRPExecutor:
    """
    Execution logic for Workstream A: VRP Operationalization
    Capital path threshold: Rs 150,000 via Half-Kelly constraint.
    """
    def __init__(self, max_capital=150000):
        self.max_capital = max_capital
        self.base_fraction = 0.275 # Half-Kelly optimal

    def check_deployment_gate(self, vix_level, current_date, expiry_date):
        """
        Task A3 - VIX and Expiry Week Gate
        Evaluates Monday open metrics to size or skip VRP execution.
        """
        days_to_expiry = (pd.to_datetime(expiry_date).date() - pd.to_datetime(current_date).date()).days
        
        # T-0 Expiry Week Rule: Skip entirely
        if 0 <= days_to_expiry <= 3: # Standard Monday -> Thursday Expiry mapping
            return 0.0, "SKIP_T0_EXPIRY_WEEK"
            
        # VIX Tiering Logic
        if vix_level < 20:
            return 1.0, "DEPLOY_FULL"
        elif 20 <= vix_level < 25:
            return 0.5, "DEPLOY_HALF"
        elif 25 <= vix_level <= 30:
            return 0.25, "DEPLOY_QUARTER"
        else:
            return 0.0, "SKIP_CRISIS_VIX_ABOVE_30"

    def calculate_strikes(self, spot_price, daily_atr):
        """
        Task A1 - Strike Selection
        NIFTY ATM logic rounding to nearest 50. Wings at ±2.5 ATR.
        """
        atm_strike = round(spot_price / 50) * 50
        wing_width = 2.5 * daily_atr
        
        long_call_strike = round((spot_price + wing_width) / 50) * 50
        long_put_strike = round((spot_price - wing_width) / 50) * 50
        
        return {
            'short_call': atm_strike,
            'short_put': atm_strike,
            'long_call': long_call_strike,
            'long_put': long_put_strike
        }

    def evaluate_exit(self, entry_net_premium, current_net_premium, current_time):
        """
        Task A2 - Target and Time Exit Calculator
        Monitors iron butterfly P&L and checks structural time exits.
        Note: Premiums represent net credit. Repurchasing closes the spread.
        """
        # Profit captured = (Sold premium) - (Current cost to buy back)
        profit_collected = entry_net_premium - current_net_premium
        max_profit = entry_net_premium
        
        # Target Exit: 50% max profit
        if profit_collected >= (0.5 * max_profit):
            return True, "TARGET_50_PERCENT"
            
        # Time Exit: Wednesday 3:00 PM (15:00)
        # weekday() -> Monday=0, Wednesday=2
        if current_time.weekday() == 2 and current_time.hour >= 15:
            return True, "TIME_EXIT_WED_1500"
            
        # Escaped Wing condition implicitly managed by structurally defined risk of the butterfly
        
        return False, "HOLD"