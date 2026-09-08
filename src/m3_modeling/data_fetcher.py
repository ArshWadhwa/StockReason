"""
Data Fetcher — Utility for downloading stock data from Yahoo Finance.
Used by the M1 pipeline (build_features.py) for data acquisition.

NOTE: This module is a standalone utility. The main data pipeline uses
src/m1_price_data/fetcher.py. This file is kept for ad-hoc data downloads.
"""
import yfinance as yf
import pandas as pd
import os


def fetch_stock_data(ticker="RELIANCE.NS", period="5y", interval="1d", output_dir="data/processed"):
    """
    Downloads OHLCV data for a single stock.
    """
    print(f"Fetching data for {ticker} (period={period}, interval={interval})...")

    os.makedirs(output_dir, exist_ok=True)

    df = yf.download(ticker, period=period, interval=interval)

    if df.empty:
        print(f"Failed to fetch data for {ticker}. Check the ticker symbol.")
        return None

    # Flatten multi-level columns if present (yfinance behavior)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    output_path = os.path.join(output_dir, f"{ticker.replace('.', '_')}.parquet")
    df.to_parquet(output_path)
    print(f"Saved data to {output_path}")

    return output_path


if __name__ == "__main__":
    fetch_stock_data()
