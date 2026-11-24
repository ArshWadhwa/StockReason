"""
Downloader with resilient retry, backoff, rate limiting, and resumable parquet caching.
Fetches up to 15 years of daily OHLCV data from yfinance for stock, index, and macro tickers.
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ws6.download")

DEFAULT_RAW_DIR = "data/v0.7/raw"
DEFAULT_LOG_PATH = os.path.join(DEFAULT_RAW_DIR, "download_log.json")


def sanitize_ticker_for_filename(ticker: str) -> str:
    """Sanitizes ticker symbols for safe filesystem usage across platforms."""
    return ticker.replace("^", "INDEX_").replace("=", "_EQ_").replace("/", "_").replace(":", "_")


def unsanitize_filename_to_ticker(filename: str) -> str:
    """Restores ticker name from sanitized filename base."""
    base = filename[:-8] if filename.endswith(".parquet") else filename
    return base.replace("INDEX_", "^").replace("_EQ_", "=")


def _load_download_log(log_path: str = DEFAULT_LOG_PATH) -> Dict[str, Any]:
    """Loads existing download log or returns empty dict."""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load download log at {log_path}: {e}")
    return {}


def _save_download_log(log_data: Dict[str, Any], log_path: str = DEFAULT_LOG_PATH) -> None:
    """Saves download log atomically to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    temp_path = f"{log_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    os.replace(temp_path, log_path)


def standardize_ohlcv_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardizes a yfinance history DataFrame into the v0.7 panel contract:
    Columns: [ticker, date, open, high, low, close, adj_close, volume]
    Keys: ticker (str), date (datetime64[ns], normalized, no timezone).
    """
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

    df = df.copy()

    # Reset index if date is in index
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # If MultiIndex columns from newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex e.g. ('Close', 'RELIANCE.NS') -> 'Close'
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Find date column
    date_col = None
    for col in df.columns:
        if str(col).lower() in ["date", "datetime", "index"]:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"No date column found in DataFrame for {ticker}")

    df = df.rename(columns={date_col: "date"})

    # Ensure date is tz-naive datetime64 and normalized
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = str(ticker).strip().upper()

    col_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    rename_dict = {col: col_mapping[col] for col in df.columns if col in col_mapping}
    df = df.rename(columns=rename_dict)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    required_cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            if c == "volume":
                df[c] = 0.0
            else:
                df[c] = np.nan

    # Enforce numeric types
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[required_cols].sort_values("date").reset_index(drop=True)
    # Drop rows where all prices are NaN
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def fetch_ticker_ohlcv(
    ticker: str,
    period: str = "15y",
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    rate_limit_delay: float = 0.05,
    cache_dir: Optional[str] = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    log_path: str = DEFAULT_LOG_PATH,
) -> Optional[pd.DataFrame]:
    """
    Fetches up to 15 years of daily OHLCV for a single ticker with retry/backoff,
    rate limiting, resumable parquet caching, and logging.

    Returns standardized DataFrame or None if unavailable.
    """
    ticker_clean = ticker.strip().upper()
    safe_name = sanitize_ticker_for_filename(ticker_clean)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        parquet_file = os.path.join(cache_dir, f"{safe_name}.parquet")
        if not force_refresh and os.path.exists(parquet_file):
            try:
                df = pd.read_parquet(parquet_file)
                # Ensure cache is not just a short snippet when multi-year data was requested
                min_expected_rows = 500 if period in ["5y", "10y", "15y"] else 5
                if not df.empty and "date" in df.columns and "close" in df.columns and len(df) >= min_expected_rows:
                    logger.debug(f"Loaded {ticker_clean} from cache: {len(df)} rows.")
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache for {ticker_clean}: {e}. Refetching...")

        # If previously recorded as failed and not force refresh, avoid slow retry loop
        if not force_refresh:
            log_data = _load_download_log(log_path)
            if log_data.get(ticker_clean, {}).get("status") == "failed":
                logger.info(f"Skipping {ticker_clean} (recorded as failed in download log).")
                return None

    # Fetch with retries and exponential backoff
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            logger.info(f"Downloading {ticker_clean} (attempt {attempt}/{max_retries}, period={period})...")
            t_obj = yf.Ticker(ticker_clean)
            raw_hist = t_obj.history(period=period, interval="1d", auto_adjust=False)

            if raw_hist is None or raw_hist.empty:
                logger.warning(f"Empty data returned for ticker {ticker_clean}")
                _update_log(log_path, ticker_clean, status="failed", error="Empty data returned", rows=0)
                return None

            df = standardize_ohlcv_dataframe(raw_hist, ticker_clean)
            if df.empty:
                logger.warning(f"Dataframe standardized to empty for ticker {ticker_clean}")
                _update_log(log_path, ticker_clean, status="failed", error="All close prices NaN", rows=0)
                return None

            # Save to cache
            if cache_dir is not None:
                df.to_parquet(parquet_file, index=False)

            _update_log(
                log_path,
                ticker_clean,
                status="success",
                rows=len(df),
                first_date=str(df["date"].min().date()),
                last_date=str(df["date"].max().date()),
            )
            return df

        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {attempt} failed for {ticker_clean}: {e}")
            if attempt < max_retries:
                sleep_time = (backoff_factor ** attempt) + np.random.uniform(0.1, 0.5)
                time.sleep(sleep_time)

    # Failed after retries
    logger.error(f"Failed to download {ticker_clean} after {max_retries} attempts: {last_err}")
    _update_log(log_path, ticker_clean, status="failed", error=str(last_err), rows=0)
    return None


def _update_log(
    log_path: str,
    ticker: str,
    status: str,
    error: Optional[str] = None,
    rows: int = 0,
    first_date: Optional[str] = None,
    last_date: Optional[str] = None,
) -> None:
    """Updates the download log entry for a ticker."""
    log_data = _load_download_log(log_path)
    log_data[ticker] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "first_date": first_date,
        "last_date": last_date,
        "error": error,
    }
    _save_download_log(log_data, log_path)


def download_universe(
    tickers: List[str],
    period: str = "15y",
    cache_dir: str = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    rate_limit_delay: float = 0.05,
    max_retries: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Resiliently downloads OHLCV data for an entire universe of tickers.
    Returns mapping from ticker to standardized DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}
    total = len(tickers)
    logger.info(f"Starting batch download for {total} tickers (period={period})...")

    for idx, ticker in enumerate(tickers, start=1):
        df = fetch_ticker_ohlcv(
            ticker=ticker,
            period=period,
            max_retries=max_retries,
            rate_limit_delay=rate_limit_delay,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        if df is not None and not df.empty:
            results[ticker] = df
        if idx % 10 == 0 or idx == total:
            logger.info(f"Progress: {idx}/{total} tickers processed ({len(results)} successful).")

    logger.info(f"Batch download complete: {len(results)}/{total} succeeded.")
    return results


def assemble_panel_dataset(
    stock_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assembles multiple stock DataFrames into a single long-format panel.
    Schema contract:
      - ticker: str
      - date: datetime64[ns]
      - open, high, low, close, adj_close, volume: float64
    """
    valid_dfs = [df for df in stock_dfs.values() if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

    panel = pd.concat(valid_dfs, ignore_index=True)
    panel["ticker"] = panel["ticker"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        panel.to_parquet(output_path, index=False)
        logger.info(f"Saved panel dataset with {len(panel)} rows to {output_path}")

    return panel
