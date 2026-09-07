import yfinance as yf
import pandas as pd
import os

def fetch_sample_data(ticker="RELIANCE.NS", start_date="2018-01-01", end_date="2024-01-01", output_dir="data/sample"):
    """
    Downloads OHLCV data for a single stock to be used as a sample for M3 prototyping.
    """
    print(f"Fetching sample data for {ticker} from {start_date} to {end_date}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print(f"Failed to fetch data for {ticker}. Check the ticker symbol.")
        return
    
    # Flatten multi-level columns if present (yfinance behavior)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{ticker.replace('.', '_')}_sample.csv")
    df.to_csv(output_path)
    print(f"Successfully saved sample data to {output_path}")
    
    return output_path

if __name__ == "__main__":
    fetch_sample_data()
