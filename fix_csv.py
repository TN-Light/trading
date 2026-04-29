import pandas as pd

try:
    df = pd.read_csv("dataset/live_options_context.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(by=['Index', 'Datetime'])
    df['Date'] = df['Datetime'].dt.date

    fixed_dfs = []
    
    # Group by Index and exact trading Date to resample the 5-minute ticks
    for (idx, date), group in df.groupby(['Index', 'Date']):
        group = group.set_index('Datetime')
        # Apply 5-minute resampling logic ONLY within the start/end window of that specific day
        resampled = group.resample('5min').ffill()
        
        # We manually add 'Date' back for concatenation, and 'Index' too
        resampled['Date'] = date
        resampled['Index'] = idx
        
        resampled = resampled.reset_index()
        fixed_dfs.append(resampled)

    final_df = pd.concat(fixed_dfs)
    
    # Sort identical to our original format (By time, then index)
    final_df = final_df.sort_values(by=['Datetime', 'Index'], ascending=[True, False])
    
    # Drop Date column
    final_df = final_df.drop(columns=['Date'])
    
    # Re-order neatly
    cols = ['Datetime', 'Index', 'Spot_Price', 'ATM_Strike', 'CE_LTP', 'PE_LTP', 'CE_OI', 'PE_OI']
    final_df = final_df[cols]
    final_df['Datetime'] = final_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    final_df.to_csv("dataset/live_options_context.csv", index=False)
    print(f"Reconstructed original rows using Forward-Fill! Dataset size now: {len(final_df)}")
    
except Exception as e:
    print(f"Error fixing tracking: {e}")