import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Stock Price Fetcher
# -----------------------------------------------------------------------------
def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch 5+ years of daily OHLCV data for a single stock ticker using yfinance.
    Returns standardized DataFrame with columns: [date, ticker, open, high, low, close, adj_close, volume].
    """
    logger.info(f"Fetching stock OHLCV data for {ticker} (period: {period})...")
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f"No data returned for ticker: {ticker}")
            return pd.DataFrame()

        df = df.reset_index()

        # Locate date column
        date_col = next((col for col in df.columns if "Date" in str(col)), None)
        if date_col:
            df = df.rename(columns={date_col: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["ticker"] = ticker.strip()

        col_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }
        df = df.rename(columns=col_mapping)

        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        required_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        for num_col in ["open", "high", "low", "close", "adj_close"]:
            if num_col in df.columns:
                df[num_col] = df[num_col].astype("float64")

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")

        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.error(f"Error fetching ticker {ticker}: {exc}")
        return pd.DataFrame()


def fetch_multiple_stocks(tickers: List[str], period: str = "5y", interval: str = "1d") -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Fetch OHLCV data for multiple stock tickers across NIFTY 50, Nifty Bank, Nifty IT, Nifty Next 50.
    Returns:
      - Combined DataFrame of valid tickers.
      - List of excluded tickers with reasons.
    """
    valid_dfs = []
    excluded_info = []

    for t in sorted(list(set(tickers))):
        df_t = fetch_stock_data(t, period=period, interval=interval)
        if df_t.empty or len(df_t) < 100:
            reason = "Insufficient history returned from source" if not df_t.empty else "No data returned from yfinance"
            logger.warning(f"Excluding ticker {t}: {reason}")
            excluded_info.append({"ticker": t, "reason": reason, "rows_fetched": len(df_t)})
        else:
            valid_dfs.append(df_t)

    if not valid_dfs:
        logger.error("No valid stock data could be fetched across requested tickers.")
        return pd.DataFrame(), excluded_info

    combined_df = pd.concat(valid_dfs, ignore_index=True)
    return combined_df, excluded_info

# -----------------------------------------------------------------------------
# 2. Market Index Fetcher (Nifty 50, Nifty Bank, Nifty IT, India VIX)
# -----------------------------------------------------------------------------
def fetch_index_data(symbol: str, name: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical daily index level data for a given index ticker.
    Returns DataFrame with columns: [date, symbol, index_name, open, high, low, close, adj_close, volume].
    """
    logger.info(f"Fetching index data for {name} ({symbol})...")
    try:
        t_obj = yf.Ticker(symbol)
        df = t_obj.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f"No data returned for index: {symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        date_col = next((col for col in df.columns if "Date" in str(col)), None)
        if date_col:
            df = df.rename(columns={date_col: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["symbol"] = symbol
        df["index_name"] = name

        col_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }
        df = df.rename(columns=col_mapping)

        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        required_cols = ["date", "symbol", "index_name", "open", "high", "low", "close", "adj_close", "volume"]
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        for num_col in ["open", "high", "low", "close", "adj_close"]:
            if num_col in df.columns:
                df[num_col] = df[num_col].astype("float64")

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")

        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.error(f"Error fetching index {symbol}: {exc}")
        return pd.DataFrame()


def fetch_all_indices(indices_dict: Dict[str, str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    results = {}
    for name, symbol in indices_dict.items():
        df_idx = fetch_index_data(symbol=symbol, name=name, period=period, interval=interval)
        if not df_idx.empty:
            results[name] = df_idx
        else:
            logger.warning(f"Failed to fetch index: {name} ({symbol})")
    return results

# -----------------------------------------------------------------------------
# 3. Macro Indicator Fetcher (USD/INR, Crude Oil)
# -----------------------------------------------------------------------------
def fetch_macro_data(symbol: str, name: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch macroeconomic daily data (e.g. USD/INR exchange rate, Crude Oil).
    """
    logger.info(f"Fetching macro data for {name} ({symbol})...")
    try:
        t_obj = yf.Ticker(symbol)
        df = t_obj.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f"No macro data returned for symbol: {symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        date_col = next((col for col in df.columns if "Date" in str(col)), None)
        if date_col:
            df = df.rename(columns={date_col: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["macro_symbol"] = symbol
        df["macro_name"] = name

        col_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }
        df = df.rename(columns=col_mapping)

        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        required_cols = ["date", "macro_symbol", "macro_name", "open", "high", "low", "close", "adj_close", "volume"]
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        for num_col in ["open", "high", "low", "close", "adj_close"]:
            if num_col in df.columns:
                df[num_col] = df[num_col].astype("float64")

        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.error(f"Error fetching macro symbol {symbol}: {exc}")
        return pd.DataFrame()


def fetch_all_macro(macro_dict: Dict[str, str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    results = {}
    for name, symbol in macro_dict.items():
        df_m = fetch_macro_data(symbol=symbol, name=name, period=period, interval=interval)
        if not df_m.empty:
            results[name] = df_m
        else:
            logger.warning(f"Failed to fetch macro indicator: {name} ({symbol})")
    return results
