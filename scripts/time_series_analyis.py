import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# --- Configuration ---
DATA_PATH = 'data/df_with_topics.pkl'
# ---------------------

def run_time_series_analysis():
    """
    Main function to execute Time Series Decomposition and Intraday Analysis.
    """
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_pickle(DATA_PATH)
    except FileNotFoundError:
        print("Error: Topic-assigned data not found. Run 02_topic_modeling.py first.")
        return

    os.makedirs('plots', exist_ok=True)

    # --- Part 1: Time Series Decomposition (Yearly Seasonality) ---
    print("\n--- Starting Time Series Decomposition (Yearly) ---")
    
    # Calculate Daily Total Article Counts
    daily_counts = df.resample('D').size().fillna(0)
    active_daily_counts = daily_counts[daily_counts > 0]

    # Use 365 days for Yearly Seasonality and the Additive Model
    try:
        decomposition = seasonal_decompose(active_daily_counts, model='additive', period=365)
        print("Time Series Decomposition successful.")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Plotting Components
        axes[0].plot(decomposition.trend, label='Trend')
        axes[0].set_title('Trend Component (Long-Term News Flow)')
        axes[1].plot(decomposition.seasonal, label='Seasonal')
        axes[1].set_title('Seasonal Component (Annual News Cycle)')
        axes[2].plot(decomposition.resid, label='Residual')
        axes[2].set_title('Irregular/Residual Component (Event Spikes)')
        axes[2].axhline(y=0, color='r', linestyle='--', linewidth=0.8)
        axes[3].plot(decomposition.observed, label='Observed')
        axes[3].set_title('Observed Daily Article Counts')
        axes[3].set_xlabel('Date')
        
        plt.suptitle('Decomposition of Financial News Publication Frequency (Yearly)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        
    except ValueError as e:
        print(f"Decomposition failed: {e}. Skipping plot.")

    # --- Part 2: Intraday Timing Analysis (Hourly Volume Profile) ---
    print("\n--- Starting Intraday Timing Analysis (Hourly Volume Profile) ---")
    
    # Extract the hour from the datetime index (assumed UTC)
    df['publication_hour_utc'] = df.index.hour
    
    # Calculate total volume per hour
    hourly_counts = df.groupby('publication_hour_utc').size()
    hourly_df = hourly_counts.to_frame(name='Total_Article_Count')

    plt.figure(figsize=(10, 6))
    hourly_df['Total_Article_Count'].plot(kind='bar', color='skyblue')

    peak_hour_utc = hourly_df['Total_Article_Count'].idxmax()
    peak_count = hourly_df['Total_Article_Count'].max()

    # Calculate estimated ET peak time (assuming UTC-5/EST)
    peak_hour_et = (peak_hour_utc - 5) % 24
    if peak_hour_et < 0: peak_hour_et += 24

    plt.title('News Publication Volume by Hour (UTC)', fontsize=16)
    plt.xlabel(f"Hour of Day (UTC) | Estimated Peak: {peak_hour_et}:00 ET", fontsize=12)
    plt.ylabel('Total Article Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Highlight the peak hour
    plt.bar(peak_hour_utc, peak_count, color='tomato', label=f'Peak Hour: {peak_hour_utc}:00 UTC')
    plt.legend()
    plt.tight_layout()
    
    print(f"Key Finding: Peak news volume occurs at {peak_hour_utc}:00 UTC, or approx. {peak_hour_et}:00 ET.")


if __name__ == '__main__':
    run_time_series_analysis()