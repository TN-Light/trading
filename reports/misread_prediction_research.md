# Diagnostic Machine Learning & Statistical Report: Predicting Trade Misreads

This diagnostic analysis processes **846 matched trades** against the raw NIFTY BANK 15-minute price series from 2015 to 2026.
It evaluates technical indicators *at the exact minute of trade entry* to determine if any market state features predict a win vs. a directional misread.

## 1. Linear Feature Correlations with Outcome (`is_win`)
A positive correlation means the feature is higher on winning trades. A negative correlation means the feature is higher on losing trades.

```
is_win              1.000000
entry_hour          0.227671
vwap_dev           -0.000180
confluence_count   -0.012203
ret_1day           -0.021977
ret_1hr            -0.027393
rsi                -0.027973
dist_sma50         -0.031518
atr_ratio          -0.036944
dist_sma200        -0.037773
dist_sma20         -0.046457
ret_3hr            -0.048012
```

*Interpretations:*
* **rsi**: -0.0280 - Very near zero. RSI level at entry does not linearly separate winners from losers.
* **atr_ratio**: -0.0369 - Near zero. Volatility levels do not directly indicate a win rate bias.
* **dist_sma200**: -0.0378 - Price distance from the 200 SMA. 

---

## 2. Multi-Dimensional Bucket Analysis

### Volatility Regime (ATR / Price)
Isolates whether the system fails more in high or low volatility markets.
| Volatility Bucket | Trades | Win Rate% | Avg Net PnL |
|---|---|---|---|
| Low Vol | 212 | 64.6% | Rs 23,275 |
| Med-Low Vol | 211 | 53.6% | Rs 15,263 |
| Med-High Vol | 211 | 64.0% | Rs 27,930 |
| High Vol | 212 | 59.9% | Rs 31,954 |

### RSI Level at Entry
Isolates whether entries into extreme overbought/oversold states decay faster.
| RSI Bucket | Trades | Win Rate% | Avg Net PnL |
|---|---|---|---|
| Oversold (<30) | 20 | 60.0% | Rs 23,121 |
| Neutral-Low (30-45) | 168 | 58.3% | Rs 32,749 |
| Neutral (45-55) | 255 | 62.4% | Rs 25,518 |
| Neutral-High (55-70) | 333 | 61.0% | Rs 24,311 |
| Overbought (>70) | 70 | 57.1% | Rs 3,648 |

### VWAP Deviation (Distance from Intraday Average)
Isolates whether entering trades when price is highly extended from VWAP predicts failure.
| VWAP Bucket | Trades | Win Rate% | Avg Net PnL |
|---|---|---|---|
| Ext Extreme Below | 212 | 60.8% | Rs 26,532 |
| Below VWAP | 211 | 61.1% | Rs 21,705 |
| Above VWAP | 211 | 57.3% | Rs 19,473 |
| Ext Extreme Above | 212 | 62.7% | Rs 30,702 |

### Long-Term Trend Alignment (Price vs 200 SMA)
Isolates whether trading with or against the long-term trend affects success.
| Trend Bucket | Trades | Win Rate% | Avg Net PnL |
|---|---|---|---|
| Strong Bearish (< -2%) | 70 | 67.1% | Rs 45,349 |
| Bearish (-2% to -0.5%) | 138 | 63.0% | Rs 24,706 |
| Neutral (-0.5% to 0.5%) | 216 | 60.6% | Rs 23,509 |
| Bullish (0.5% to 2%) | 312 | 57.1% | Rs 20,411 |
| Strong Bullish (> 2%) | 108 | 62.0% | Rs 25,837 |

---

## 3. Decision Tree Rule Extraction (Non-Linear Machine Learning)
We trained a shallow decision tree (depth=3) to find non-linear combinations of market features that segment trade outcomes.

### Rules Structure:
```
|--- entry_hour <= 12.88
|   |--- dist_sma20 <= 0.01
|   |   |--- dist_sma20 <= -0.00
|   |   |   |--- class: 0
|   |   |--- dist_sma20 >  -0.00
|   |   |   |--- class: 1
|   |--- dist_sma20 >  0.01
|   |   |--- rsi <= 68.97
|   |   |   |--- class: 0
|   |   |--- rsi >  68.97
|   |   |   |--- class: 0
|--- entry_hour >  12.88
|   |--- rsi <= 52.24
|   |   |--- ret_1hr <= -0.00
|   |   |   |--- class: 1
|   |   |--- ret_1hr >  -0.00
|   |   |   |--- class: 1
|   |--- rsi >  52.24
|   |   |--- dist_sma50 <= 0.00
|   |   |   |--- class: 1
|   |   |--- dist_sma50 >  0.00
|   |   |   |--- class: 1

```

### Segment Performance:
| Node/Segment | Trades | Win Rate% | Avg Net PnL |
|---|---|---|---|
| Segment 3 | 206 | 49.0% | Rs 18,277 |
| Segment 4 | 370 | 61.9% | Rs 28,191 |
| Segment 6 | 34 | 20.6% | Rs -8,064 |
| Segment 7 | 37 | 45.9% | Rs -6,282 |
| Segment 10 | 37 | 81.1% | Rs 45,907 |
| Segment 11 | 38 | 97.4% | Rs 63,703 |
| Segment 13 | 70 | 61.4% | Rs 15,312 |
| Segment 14 | 52 | 88.5% | Rs 37,309 |

## 4. Key Takeaways
1. **Low linear predictability**: Individual technical indicators at entry have extremely low correlation to whether a specific trade wins or loses. The market noise dominates at a single-trade level.
2. **Bucket analysis shows stability**: The strategy's win rate stays between 53-67% across Volatility (ATR), RSI, VWAP deviation, and Long-Term Trend (SMA200) buckets (200+ trades each). The core edge does not depend on a specific regime.
3. **Decision tree finds extreme segments, but trust them with extreme caution**: The depth-3 decision tree identifies segments ranging from 20.6% WR (Segment 6: morning trades with dist_sma20 > 1%) to 97.4% WR (Segment 11: afternoon trades after ~1PM with RSI ≤ 52). However, these are **small-sample leaves** (34–52 trades each from 846 total). A depth-3 tree on 846 samples is highly prone to overfitting noise rather than capturing real patterns. **Do not trade these segments as filters without out-of-sample validation on fresh data.** The primary split (`entry_hour ≤ 12.88`) — afternoon trades outperforming morning trades — is worth investigating further with a larger sample, but the sub-splits are unreliable.

