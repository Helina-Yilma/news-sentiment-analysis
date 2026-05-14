import pandas as pd
import numpy as np
import talib 
import pynance as pn 
import os

# --- Configuration ---
data_path = 'data/yfinance_data/'
FILES = ['aapl.csv', 'amzn.csv', 'goog.csv', 'meta.csv', 'msft.csv', 'nvda.csv']
# ---------------------

def load_data(folder_path, files):
    """Loads all CSV files, cleans data, and combines them into one DataFrame."""
    all_data = []
    
    print(f"Loading data from: {folder_path}")
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        try:
            ticker = file.split('.')[0].upper()
            df = pd.read_csv(file_path) 
            df['Ticker'] = ticker
            
            # Data Cleaning and Preparation
            df['Date'] = pd.to_datetime(df['Date'])
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in required_cols:
                if col not in df.columns: raise ValueError(f"Missing column '{col}'")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            all_data.append(df.dropna(subset=required_cols)) 
            
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}. Skipping.")
        except ValueError as e:
            print(f"ERROR in {file}: {e}. Skipping.")

    if not all_data:
        raise RuntimeError("No valid data loaded. Check paths and file integrity.")
        
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.reset_index(drop=True)
    df_combined = df_combined.set_index(['Ticker', 'Date']).sort_index()
    print("Data loading complete.")
    return df_combined

def calculate_technical_indicators(df):
    """Calculates SMA, RSI, and MACD using TA-Lib."""
    print("Calculating Technical Indicators (TA-Lib)...")

    def apply_talib(group):
        close = group['Close'].values

        # TA-Lib Indicators
        group['SMA_20'] = talib.SMA(close, timeperiod=20)
        group['RSI_14'] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        group['MACD'] = macd
        group['MACD_Signal'] = macd_signal
        group['MACD_Hist'] = macd_hist
        
        return group

    return df.groupby('Ticker', group_keys=False).apply(apply_talib)

def calculate_financial_metrics(df):
    """Calculates financial metrics (Returns, Volatility) using PyNance."""
    print("Calculating Financial Metrics (PyNance)...")

    def apply_pynance(group):
        close_series = group['Close']
        
        # PyNance Functions
        # Daily Returns (percentage change)
        group['Daily_Return'] = pn.returns(close_series)
        
        # Cumulative Returns (total growth)
        group['Cumulative_Return'] = pn.cum_returns(close_series) 
        
        # Volatility (annualized 20-day rolling standard deviation)
        group['Volatility_20d'] = pn.rolling_vol(close_series, window=20) 
        
        return group

    return df.groupby('Ticker', group_keys=False).apply(apply_pynance)


if __name__ == '__main__':
    try:
        df_processed = load_data(data_path, FILES)
        df_processed = calculate_technical_indicators(df_processed)
        df_processed = calculate_financial_metrics(df_processed)
        
        print("\n--- Final Processed Data Head ---")
        print(df_processed.head(10))
        
        # Save the result so it can be used by the visualization script
        df_processed.to_csv('processed_stock_analysis.csv')
        print("\ Processed data saved to 'processed_stock_analysis.csv'")

    except RuntimeError as e:
        print(f"\nFATAL ERROR: {e}")