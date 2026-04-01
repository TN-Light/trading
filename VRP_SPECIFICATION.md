# VRP System Specification: Iron Butterfly & T-0 Rules

## Iron Butterfly Strategy Rules
1. **Entry Setup**: Define At-The-Money (ATM) strike using the current underlying price at a fixed time each trading session.
2. **Legs Definition**:
   - Sell 1 ATM Call
   - Sell 1 ATM Put
   - Buy 1 Out-of-The-Money (OTM) Call
   - Buy 1 Out-of-The-Money (OTM) Put
   - (The OTM wings provide tail risk protection and synthetically cap max loss).
3. **Width Specifications**: The wing width varies dynamically by delta (e.g., 10-delta or 15-delta equivalents) or using a fixed static-width based on the underlying implied volatility (VIX).
4. **Risk Management Configuration**: 
   - Profit cap is generally structured at a dynamic percentage of net premium collected (e.g., Target 20-30% of total credit).
   - Stop-Loss is bounded by the max threshold defined on the premium side or a strict 1x to 1.5x of the credit received.

## T-0 Skip Rule (Expiry Day / Week Rule)
1. **Definition**: The "T-0" rule dictates that standard multi-day iron butterfly deployments **must be skipped** on the exact week of expiration (0 DTE week) for the chosen underlying/index.
2. **Rationale**: Gamma and Theta interactions are extremely non-linear in their terminal phase. Expiry 0-DTE movements behave erratically, yielding negative expected value within the core mean-reversion VRP parameters.
3. **Execution Logic**: 
   - **IF** the current week is the SEBI monthly expiry week.
   - **THEN** Force action to `SKIP_TRADE` or force reallocation to the next sequential cycle.
   - This explicitly bypasses zero-DTE "pin risk" scenarios and gamma-driven degradation.
