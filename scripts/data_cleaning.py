import pandas as pd
import re
import os
import datetime as dt

# --- Configuration ---
RAW_DATA_PATH = 'data/newsData/raw_analyst_ratings.csv'
CLEANED_DATA_PATH = 'data/df_cleaned.pkl'
# ---------------------

def extract_domain(publisher_name):
    """
    Extracts a clean domain name from the publisher column, handling email formats.
    """
    publisher_name = str(publisher_name).lower().strip()
    if '@' in publisher_name:
        match = re.search(r'@(.+?)(?:\.\w+)?$', publisher_name)
        if match:
            # Return the domain part before the TLD (e.g., 'benzinga' from '@benzinga.com')
            return match.group(1)
    # If not an email, clean up the string (e.g., remove spaces)
    return publisher_name.replace(' ', '_')

def run_data_cleaning():
    """
    Main function to load, clean, and save the base DataFrame.
    """
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
    df = pd.DataFrame(data).set_index('timestamp')
    df.index = pd.to_datetime(df.index, utc=True)
        
    print(f"Raw data shape: {df.shape}")
    
    # 1. Ensure the index is a proper datetime object (crucial for time analysis)
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Converting index to datetime...")
        df.index = pd.to_datetime(df.index, utc=True)
    
    # 2. Drop rows with missing headlines or publishers
    df.dropna(subset=['headline', 'publisher'], inplace=True)
    
    # 3. Create the normalized 'publisher_domain' column
    df['publisher_domain'] = df['publisher'].apply(extract_domain)
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    
    # 4. Save the cleaned DataFrame
    df.to_pickle(CLEANED_DATA_PATH)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}. Final shape: {df.shape}")

if __name__ == '__main__':
    run_data_cleaning()