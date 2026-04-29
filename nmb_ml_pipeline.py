import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def run_nmb_ml_pipeline(csv_path='dataset/nmb_ml_training_data.csv', probability_threshold=0.50):
    print("=======================================================")
    print(" NMB ENGINE: XGBoost Probability Filter Protocol")
    print("=======================================================\n")
    
    # 1. Load the generated dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Run the data generator first.")
        return
        
    if len(df) < 50:
        print("Not enough data points to train a robust model. Loosen engine constraints further.")
        return

    # 2. Feature Selection (X) and Target (y)
    # Dropping non-predictive columns and the target variables
    features = ['box_width', 'volume_surge_ratio', 'day_of_week', 'dir', 'gap_pct', 'atr_ratio', 'vwap_dist_pct']
    X = df[features]
    y = df['target_hit']  # 1 for Hit Target, 0 for Stop Loss / Time Stop
    
    # 3. Chronological Train/Test Split (Crucial for Time-Series)
    # We train on the first 75% of the timeline, test on the unseen final 25%
    split_idx = int(len(df) * 0.75)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    net_pts_test = df['net_pts'].iloc[split_idx:] # Keep track of the actual points for PnL
    
    print(f"Total Trades in Dataset : {len(df)}")
    print(f"Training Set (Older)    : {len(X_train)} trades")
    print(f"Testing Set (Unseen)    : {len(X_test)} trades\n")

    # 4. Initialize and Train the XGBoost Classifier
    # Tuned to prevent overfitting on noisy financial data
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,            # Shallow trees to prevent overfitting the noise
        learning_rate=0.05,
        subsample=0.8,          # Randomly sample 80% of data per tree
        colsample_bytree=0.8,   # Randomly sample 80% of features per tree
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training XGBoost Model...")
    model.fit(X_train, y_train)
    
    # 5. Predict Probabilities on the Unseen Test Data
    # predict_proba returns [Probability of 0, Probability of 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 6. Apply the Strict Execution Blocker (Threshold > 0.65)
    # Only take the trade if the ML model is > 65% confident it will hit the target
    y_pred_custom = (y_pred_proba >= probability_threshold).astype(int)
    
    # 7. Evaluate the Financial Impact
    # Compare the "Dumb" Base Engine vs the "Smart" ML-Filtered Engine
    
    # Base Engine Metrics (Taking every trade in the test set)
    base_wins = y_test.sum()
    base_total = len(y_test)
    base_win_rate = (base_wins / base_total) * 100
    base_pnl = net_pts_test.sum() * 15 # Rs PnL proxy
    
    # ML Filtered Metrics (Only taking trades that passed the threshold)
    # Mask where our model said "YES"
    taken_trades_mask = y_pred_custom == 1
    
    if taken_trades_mask.sum() == 0:
        print(f"WARNING: The model found NO trades that met the {probability_threshold*100}% confidence threshold.")
        print("This means capital is 100% safe, but the machine is too strict. Consider lowering threshold to 0.60.")
        return

    ml_wins = y_test[taken_trades_mask].sum()
    ml_total = taken_trades_mask.sum()
    ml_win_rate = (ml_wins / ml_total) * 100
    ml_pnl = net_pts_test[taken_trades_mask].sum() * 15 # Rs PnL proxy
    trades_skipped = base_total - ml_total
    
    print("=======================================================")
    print(" OUT-OF-SAMPLE TEST RESULTS (Unseen Data Only)")
    print("=======================================================")
    print(f" [BASE ENGINE - NO ML]")
    print(f" Trades Taken : {base_total}")
    print(f" Win Rate     : {base_win_rate:.2f}%")
    print(f" Net PnL (Rs) : {base_pnl:,.2f}")
    print("-" * 55)
    print(f" [NEXUS ML ENGINE - {probability_threshold*100}% THRESHOLD]")
    print(f" Trades Taken : {ml_total} (Skipped {trades_skipped} bad setups)")
    print(f" Win Rate     : {ml_win_rate:.2f}%")
    print(f" Net PnL (Rs) : {ml_pnl:,.2f}")
    print("=======================================================\n")
    
    # Show Feature Importance
    importance = model.feature_importances_
    print(" Feature Importance driving the AI:")
    for feat, imp in zip(features, importance):
        print(f" - {feat:<20}: {imp*100:.1f}%")

if __name__ == '__main__':
    run_nmb_ml_pipeline()