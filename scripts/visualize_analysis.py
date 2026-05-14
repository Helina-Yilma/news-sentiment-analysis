import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display

# --- Configuration ---
PROCESSED_DATA_FILE = 'processed_stock_analysis.csv'
# ---------------------

def visualize_comparative_analysis(df):
    """Generates comparative plots for all companies."""
    print("\n--- Generating Comparative Visualization ---")

    # Reset index for easy plotting with Seaborn 'hue'
    df_plot = df.reset_index()
    all_tickers = df_plot['Ticker'].unique()
    
    sns.set_style("whitegrid")
    
    # Create figure with two subplots: Price and Cumulative Returns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"Comparative Financial and Price Analysis for {len(all_tickers)} Tech Stocks", fontsize=16)

    # --- AXIS 1: Comparative Close Prices ---
    sns.lineplot(data=df_plot, x='Date', y='Close', hue='Ticker', ax=ax1, linewidth=1.5)
    
    ax1.set_title('Comparative Close Prices Over Time')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left', title='Ticker')

    # --- AXIS 2: Comparative Cumulative Returns ---
    sns.lineplot(data=df_plot, x='Date', y='Cumulative_Return', hue='Ticker', ax=ax2, linewidth=1.5, legend=False)
    
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax2.set_title('Comparative Cumulative Returns')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_xlabel('Date')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def display_financial_summary(df):
    """Displays key financial metrics across all companies."""
    
    # Calculate Mean Annualized Volatility and Total Cumulative Return
    summary = df.groupby('Ticker').agg(
        Mean_Volatility_20d=('Volatility_20d', 'mean'),
        Total_Cumulative_Return=('Cumulative_Return', 'last')
    ).reset_index()
    
    print("\n--- Financial Summary (Mean Volatility & Total Return) ---")
    display(summary.style.format({
        'Mean_Volatility_20d': "{:.2%}",
        'Total_Cumulative_Return': "{:.2%}"
    }))

if __name__ == '__main__':
    try:
        # Load the processed data with the MultiIndex and ensure Date is parsed correctly
        df_analysis = pd.read_csv(PROCESSED_DATA_FILE, index_col=['Ticker', 'Date'], parse_dates=['Date'])
        
        display_financial_summary(df_analysis)
        visualize_comparative_analysis(df_analysis)
        
    except FileNotFoundError:
        print(f"ERROR: Processed data file '{PROCESSED_DATA_FILE}' not found. Run 'load_and_process_data.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred during visualization: {e}")