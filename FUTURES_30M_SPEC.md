# 30-Minute NIFTY Futures Classifier Specification

## 1. Theoretical Foundation
This specification targets the microstructure footprint of institutional order flow at the 30-minute horizon. At 5-minute resolutions, High-Frequency Trading (HFT) and algorithmic latency arbitrage effectively clear inefficiencies, perfectly validating Fama's semi-strong EMH. 

However, at 30 minutes, two physiological/regulatory constraints of the market emerge:
1. **Order Flow Gestation:** A ₹500 crore institutional block cannot be executed in 5 minutes without causing severe market impact. Execution spans 4 to 8 consecutive 30-minute intervals. The footprint of this activity (Easley and O'Hara, 1987) becomes a measurable correlation.
2. **SEBI Margin Reporting Checkpoints:** Indian exchanges enforce strict intraday margin reporting (predominantly 11:30 AM, 1:00 PM, and 2:30 PM), forcing mandatory risk adjustments and algorithmic rebalancing that create a regulatory-driven directional bias as opposed to an informational one.

## 2. Source Data Requirements
- **Instrument**: NIFTY Near-Month Continuous Futures (Panama Canal roll-adjusted ratio).
- **Timeframe**: 30-minute exact intervals (e.g. 09:15-09:45).
- **Fields**: Open, High, Low, Close, Volume, Open Interest (OI), Cash Index Close.

## 3. Mathematical Feature Engineering Specification

### Phase 1: Amihud Illiquidity (Absorption vs. Initiation)
**Theory**: Kyle's Lambda / Amihud (2002). Captures whether price movement is happening due to thin liquidity or genuine institutional commitment.
**Formula**:
- **Raw Amihud ($A_t$)**: $A_t = \frac{|Return_t|}{Volume_t}$
- **Normalized Illiquidity ($\hat{A}_t$)**: $\hat{A}_t = \frac{A_t}{SMA(A_{t}, 20)}$
*Implementation Note*: Compute on 30-minute bars. Values near 0 indicate massive volume absorbed the move (strongest conviction), values $>> 1$ indicate price gaps on thin liquidity.

### Phase 2: Open Interest (OI) Rate of Change
**Theory**: Bessembinder and Seguin (1993). Rising OI + Rising Price = Genuine Longs. Falling OI + Rising Price = Short Covering. 
**Formula**:
- **Mean-Stabilized OI ($\hat{OI}_t$)**: Since OI climbs globally as a new contract begins and drops toward expiry, raw levels are useless.
- $\hat{OI}_t = \frac{OI_t}{SMA(OI_{t}, 20)}$
- **OI Momentum ($dOI/dt$)**: $dOI_t = \hat{OI}_t - \hat{OI}_{t-1}$
*Implementation Note*: Track $dOI_t$ alongside the sign of the 30-minute return. 

### Phase 3: The Basis Signature (Futures vs. Cash)
**Theory**: Premium/Discount encodes dividend expectations, funding costs, and extreme short-term institutional speculation.
**Formula**:
- **Raw Basis ($B_t$)**: $B_t = (Futures Close_t - Cash Close_t)$
- **Basis Premium Percentage**: $BP_t = \frac{B_t}{Cash Close_t} \times 100$
*Implementation Note*: Extreme deviations in $BP_t$ often precede mean-reverting shocks at the 30-minute horizon.

### Phase 4: Biomimetic Derivative Features (Velocity)
**Theory**: Echolocation doesn't measure distance, it measures Doppler shift (velocity). We must measure the *rate of state change*, not the static state.
**Formula**:
- **Illiquidity Velocity**: $\Delta\hat{A}_t = \hat{A}_t - \hat{A}_{t-1}$ 
- **OI Acceleration**: $d^2OI_t = dOI_t - dOI_{t-1}$
- **Basis Contraction/Expansion**: $\Delta BP_t = BP_t - BP_{t-1}$
*Implementation Note*: A narrowing basis ($\Delta BP_t < 0$) coupled with expanding volume velocity is the actual signal of an impending break, rather than static threshold crossing.

## 4. Evaluation Branches
If the empirical data on 60-days of Angel One futures supports autocorrelation `> 0.05` in Q1 (Lowest Amihud), this framework is coded exactly as specified. 

If it fails, this domain (price/volume/OI) is officially exhausted under EMH, pivoting strategy to purely exogenous data (FII/DII End-of-Day prints, Option Chain Gamma walls, or higher daily horizons).
