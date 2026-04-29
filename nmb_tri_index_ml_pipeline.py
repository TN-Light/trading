import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def run_multi_model_pipeline(probability_threshold=0.65):
    indices = ['NIFTY_BANK', 'NIFTY_50', 'FINNIFTY']
    os.makedirs('models', exist_ok=True)
    
    features = [
        'box_width', 'dir', 'gap_pct', 'atr_ratio', 'vwap_dist_pct', 
        'anchor_1_vwap_div', 'anchor_2_vwap_div', 'straddle_premium', 'pcr'
    ]

    for symbol in indices:
        csv_path = f'dataset/{symbol}_ml_training_data.csv'
        print(f"\n=======================================================")
        print(f" TRAINING XGBOOST MODEL: {symbol}")
        print(f"=======================================================")
        
        try:
            df = pd.read_csv(csv_path)
            # XGBoost automatically handles NaNs, but we must drop NaNs from structural spot features
            core_spot_features = [f for f in features if f not in ['straddle_premium', 'pcr']]
            df = df.dropna(subset=core_spot_features)
        except FileNotFoundError:
            print(f"Error: Could not find {csv_path}. Run the engine first.")
            continue

        if len(df) < 50:
            print(f"Not enough data points ({len(df)}) to train {symbol}.")
            continue

        X = df[features]
        y = df['target_hit']

        split_idx = int(len(df) * 0.75)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        net_pts_test = df['net_pts'].iloc[split_idx:]

        print(f"Total Trades : {len(df)}")
        print(f"Training Set : {len(X_train)} trades")
        print(f"Testing Set  : {len(X_test)} trades\n")

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        print("Fitting model...")
        model.fit(X_train, y_train)

        model_filename = f'models/{symbol}_nmb_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Successfully exported model to {model_filename}")

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_custom = (y_pred_proba >= probability_threshold).astype(int)

        base_total = len(y_test)
        base_win_rate = (y_test.sum() / base_total) * 100 if base_total > 0 else 0
        base_pnl = net_pts_test.sum() * 15

        taken_trades_mask = (y_pred_custom == 1)
        ml_total = taken_trades_mask.sum()
        
        if ml_total == 0:
            print(f"WARNING: No test trades passed the {probability_threshold*100}% threshold.")
            print("Confidence is strictly zero. Retraining needed or edge missing for this threshold.\n")
        else:
            ml_win_rate = (y_test[taken_trades_mask].sum() / ml_total) * 100
            ml_pnl = net_pts_test[taken_trades_mask].sum() * 15
            trades_skipped = base_total - ml_total

            print(f" [BASE ENGINE - NO ML]")
            print(f" Trades: {base_total} | WR: {base_win_rate:.2f}% | PnL: Rs {base_pnl:,.2f}")
            print(f" [ML FILTERED - THRESHOLD: {probability_threshold*100}%]")
            print(f" Trades: {ml_total} (Skipped {trades_skipped}) | WR: {ml_win_rate:.2f}% | PnL: Rs {ml_pnl:,.2f}")

        print("\n Feature Importance:")
        for feat, imp in zip(features, model.feature_importances_):
            print(f" - {feat:<20}: {imp*100:.1f}%")

if __name__ == '__main__':
    run_multi_model_pipeline(probability_threshold=0.65)
