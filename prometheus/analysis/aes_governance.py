# ============================================================================
# PROMETHEUS APEX — AES Factor Decomposition Governance
# ============================================================================
"""
Provides a stable, transparent pipeline to flatten and log exactly how much each quant 
pillar contributed to an entry decision. Translates the AES factor decomposition into 
a diagnostic structure that solves the parameter-interaction blackbox.
"""

from typing import Dict, Any

class AesGovernanceLog:
    """
    Standardized payload formatting for AES metrics tracking per trade.
    """
    
    @staticmethod
    def flatten_trade_features(
        trade_id: str,
        symbol: str, 
        edge_score: int, 
        sizing_tier: str,
        aes_components: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Parses the AES decomposition dictionary into a flat row for CSV/DB logging.
        """
        payload = {
            "trade_id": trade_id,
            "symbol": symbol,
            "aes_total_score": edge_score,
            "sizing_tier": sizing_tier,
        }
        
        # Safe extraction of the 6 core pillars of conviction
        payload["aes_regime_alignment"] = round(aes_components.get("regime_alignment", 0.0), 2)
        payload["aes_signal_confluence"] = round(aes_components.get("signal_confluence", 0.0), 2)
        payload["aes_volatility_support"] = round(aes_components.get("volatility_support", 0.0), 2)
        payload["aes_gravity_clearance"] = round(aes_components.get("gravity_clearance", 0.0), 2)
        payload["aes_time_decay_edge"] = round(aes_components.get("time_decay_edge", 0.0), 2)
        payload["aes_macro_flow"] = round(aes_components.get("macro_flow", 0.0), 2)
        
        return payload

    @staticmethod
    def inject_to_dataframe(df, aes_log_list: list):
        """
        Takes a list of flattened feature dicts and merges them safely 
        into the main trade analysis dataframe.
        """
        import pandas as pd
        
        if not aes_log_list:
            return df
            
        aes_df = pd.DataFrame(aes_log_list)
        
        # Merge on trade_id
        if "trade_id" in df.columns:
            return pd.merge(df, aes_df, on="trade_id", how="left")
            
        return df

