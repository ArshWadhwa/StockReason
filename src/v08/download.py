"""
StockReason v0.8 Resilient Historical OHLCV Downloader.
=====================================================
Acquires daily OHLCV + Adjusted Close market data from 2011-11-01 (15+ years)
to the latest completed NSE trading session with:
- Checkpointing and resumable incremental refresh.
- Exponential backoff and rate limiting.
- Atomic parquet storage.
- Explicit failure tracking (never treats empty dataset as success).
- Source provenance recording with SHA-256 integrity verification.
"""

import os
import sys
import time
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

from src.v08.contracts import validate_ohlcv_dataframe, REQUIRED_OHLCV_COLUMNS

logger = logging.getLogger("v08.download")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_RAW_PRICES_DIR = "data/v0.8/raw/prices"
DEFAULT_CHECKPOINT_PATH = "data/v0.8/metadata/download_checkpoint.json"
DEFAULT_PROVENANCE_PATH = "data/v0.8/metadata/provenance.json"
DEFAULT_START_DATE = "2011-11-01"


def sanitize_ticker(ticker: str) -> str:
    """Sanitizes ticker symbols for cross-platform filesystem filenames."""
    return ticker.replace("^", "INDEX_").replace("=", "_EQ_").replace("/", "_").replace(":", "_")


def unsanitize_ticker(filename: str) -> str:
    """Reverses sanitized filename base back to ticker symbol."""
    base = filename[:-8] if filename.endswith(".parquet") else filename
    return base.replace("INDEX_", "^").replace("_EQ_", "=")


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for integrity provenance."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_json_record(path: str) -> Dict[str, Any]:
    """Loads a JSON file or returns an empty dict if missing/corrupt."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading JSON from {path}: {e}")
    return {}


def save_json_atomic(data: Dict[str, Any], path: str) -> None:
    """Saves a dictionary to JSON atomically using a temporary file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp_{int(time.time() * 1000)}"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(temp_path, path)


def standardize_ohlcv_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardizes a downloaded OHLCV DataFrame into the v0.8 contract:
    Columns: [ticker, date, open, high, low, close, adj_close, volume]
    Date normalized to tz-naive datetime64 at midnight.
    """
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)

    df = df.copy()

    # Reset index if date is stored in index
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Flatten MultiIndex columns if present from recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Identify date column
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "datetime", "index"]:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"No date column found in downloaded data for {ticker}")

    df = df.rename(columns={date_col: "date"})

    # Normalize date to tz-naive datetime64[ns] at midnight
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    df["ticker"] = str(ticker).strip().upper()

    # Column mapping
    mapping = {
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
        "adj_close": "adj_close",
        "volume": "volume",
    }
    df = df.rename(columns=mapping)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Retain strictly required columns
    df = df[REQUIRED_OHLCV_COLUMNS]

    # Type casting
    numeric_cols = ["open", "high", "low", "close", "adj_close"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)

    # Sort, deduplicate, and drop all-NaN prices
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_ticker_ohlcv(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    output_dir: str = DEFAULT_RAW_PRICES_DIR,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    provenance_path: str = DEFAULT_PROVENANCE_PATH,
    max_retries: int = 4,
    backoff_factor: float = 1.5,
    rate_limit_delay: float = 0.1,
    incremental: bool = True,
    v07_fallback_dir: str = "data/v0.7/raw",
) -> Dict[str, Any]:
    """
    Fetches daily OHLCV data for a ticker with caching, retry, and incremental refresh.
    Returns status dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    sanitized = sanitize_ticker(ticker)
    out_file = os.path.join(output_dir, f"{sanitized}.parquet")

    now_utc = datetime.now(timezone.utc)
    if end_date is None:
        end_date = now_utc.strftime("%Y-%m-%d")

    df_existing = None
    query_start = start_date

    # 1. Check existing parquet file for incremental refresh
    if incremental and os.path.exists(out_file):
        try:
            df_existing = pd.read_parquet(out_file)
            if not df_existing.empty and "date" in df_existing.columns:
                df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.tz_localize(None).dt.normalize()
                max_date = df_existing["date"].max()
                earliest_date = df_existing["date"].min()

                # If existing data already starts <= start_date and ends >= end_date (or latest session within 2 days)
                target_end_dt = pd.to_datetime(end_date)
                target_start_dt = pd.to_datetime(start_date)

                # Check if up to date allowing for weekend / holiday calendar gap (up to 4 calendar days)
                cal_gap_days = (target_end_dt - max_date).days
                if earliest_date <= target_start_dt and cal_gap_days <= 4:
                    logger.debug(f"{ticker}: Up to date through {max_date.strftime('%Y-%m-%d')} (gap {cal_gap_days}d). Cached.")
                    return {
                        "ticker": ticker,
                        "status": "cached",
                        "rows": len(df_existing),
                        "first_date": earliest_date.strftime("%Y-%m-%d"),
                        "last_date": max_date.strftime("%Y-%m-%d"),
                        "filepath": out_file,
                        "error": None,
                    }

                # If missing newer days, query from max_date + 1 day
                if earliest_date <= target_start_dt and max_date < target_end_dt:
                    next_day = max_date + pd.Timedelta(days=1)
                    query_start = next_day.strftime("%Y-%m-%d")
                    logger.debug(f"{ticker}: Incremental query from {query_start} to {end_date}")
        except Exception as e:
            logger.warning(f"Could not read existing parquet for {ticker} ({e}). Performing full query.")
            df_existing = None
            query_start = start_date

    # 2. Resilient download loop with exponential backoff
    attempt = 0
    df_downloaded = None
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            time.sleep(rate_limit_delay)
            yf_ticker = yf.Ticker(ticker)
            raw_data = yf_ticker.history(
                start=query_start,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )

            if raw_data is not None and not raw_data.empty:
                df_downloaded = standardize_ohlcv_dataframe(raw_data, ticker)
                break
            else:
                last_error = "Provider returned empty dataset"
                # If incremental query returned empty, market may not have traded new days
                if df_existing is not None and not df_existing.empty:
                    df_downloaded = pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)
                    break
        except Exception as e:
            last_error = str(e)
            sleep_time = backoff_factor ** attempt
            logger.warning(f"{ticker}: attempt {attempt}/{max_retries} failed ({e}). Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # 3. Fallback to v0.7 archive if yfinance query failed completely and existing is empty
    if (df_downloaded is None or df_downloaded.empty) and (df_existing is None or df_existing.empty):
        v07_file = os.path.join(v07_fallback_dir, f"{sanitized}.parquet")
        if os.path.exists(v07_file):
            try:
                v07_df = pd.read_parquet(v07_file)
                if not v07_df.empty:
                    logger.info(f"{ticker}: Bridging from v0.7 archive ({len(v07_df)} rows).")
                    df_downloaded = standardize_ohlcv_dataframe(v07_df, ticker)
            except Exception as e:
                logger.warning(f"{ticker}: v0.7 archive fallback failed: {e}")

    # 4. Merge downloaded data with existing data if incremental
    final_df = None
    if df_existing is not None and not df_existing.empty:
        if df_downloaded is not None and not df_downloaded.empty:
            combined = pd.concat([df_existing, df_downloaded], ignore_index=True)
            final_df = combined.drop_duplicates(subset=["ticker", "date"]).sort_values("date").reset_index(drop=True)
        else:
            final_df = df_existing
    else:
        final_df = df_downloaded

    # 5. Handle complete failure
    if final_df is None or final_df.empty:
        err_msg = last_error or "Empty dataset after retries"
        logger.warning(f"{ticker}: Download failed — {err_msg}")
        return {
            "ticker": ticker,
            "status": "failed",
            "rows": 0,
            "first_date": None,
            "last_date": None,
            "filepath": None,
            "error": err_msg,
        }

    # 6. Validate final dataset
    is_valid, validation_errors = validate_ohlcv_dataframe(final_df, allow_empty=False)
    if not is_valid:
        err_msg = f"Contract validation errors: {validation_errors}"
        logger.error(f"{ticker}: {err_msg}")
        return {
            "ticker": ticker,
            "status": "validation_failed",
            "rows": len(final_df),
            "first_date": None,
            "last_date": None,
            "filepath": None,
            "error": err_msg,
        }

    # 7. Atomic Parquet Write
    temp_file = f"{out_file}.tmp_{int(time.time() * 1000)}"
    final_df.to_parquet(temp_file, index=False, compression="snappy")
    os.replace(temp_file, out_file)

    first_dt_str = final_df["date"].min().strftime("%Y-%m-%d")
    last_dt_str = final_df["date"].max().strftime("%Y-%m-%d")
    file_hash = compute_file_sha256(out_file)

    # 8. Record Checkpoint and Provenance
    checkpoint_data = load_json_record(checkpoint_path)
    checkpoint_data[ticker] = {
        "status": "success",
        "timestamp": now_utc.isoformat(),
        "rows": len(final_df),
        "first_date": first_dt_str,
        "last_date": last_dt_str,
        "sha256": file_hash,
        "error": None,
    }
    save_json_atomic(checkpoint_data, checkpoint_path)

    provenance_data = load_json_record(provenance_path)
    provenance_data[ticker] = {
        "provider": "yfinance_nse",
        "retrieved_utc": now_utc.isoformat(),
        "query_start": start_date,
        "query_end": end_date,
        "first_trading_date": first_dt_str,
        "latest_trading_date": last_dt_str,
        "record_count": len(final_df),
        "sha256": file_hash,
    }
    save_json_atomic(provenance_data, provenance_path)

    logger.info(f"{ticker}: Saved {len(final_df)} daily observations ({first_dt_str} to {last_dt_str}).")
    return {
        "ticker": ticker,
        "status": "success",
        "rows": len(final_df),
        "first_date": first_dt_str,
        "last_date": last_dt_str,
        "filepath": out_file,
        "sha256": file_hash,
        "error": None,
    }


def download_universe_ohlcv(
    tickers: List[str],
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    output_dir: str = DEFAULT_RAW_PRICES_DIR,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    provenance_path: str = DEFAULT_PROVENANCE_PATH,
    rate_limit_delay: float = 0.1,
    incremental: bool = True,
) -> Dict[str, Any]:
    """
    Downloads OHLCV data for an entire universe of tickers.
    Tracks success, failure, rows, and coverage summary.
    """
    logger.info(f"Starting OHLCV acquisition for {len(tickers)} securities from {start_date}...")
    start_time = time.time()
    results = {}
    success_cnt = 0
    cached_cnt = 0
    fail_cnt = 0
    total_rows = 0

    for idx, ticker in enumerate(tickers, start=1):
        res = fetch_ticker_ohlcv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            provenance_path=provenance_path,
            rate_limit_delay=rate_limit_delay,
            incremental=incremental,
        )
        results[ticker] = res
        status = res["status"]
        if status == "success":
            success_cnt += 1
            total_rows += res["rows"]
        elif status == "cached":
            cached_cnt += 1
            total_rows += res["rows"]
        else:
            fail_cnt += 1

        if idx % 25 == 0 or idx == len(tickers):
            logger.info(f"Progress: [{idx}/{len(tickers)}] | Success: {success_cnt} | Cached: {cached_cnt} | Failed: {fail_cnt}")

    elapsed = time.time() - start_time
    summary = {
        "total_tickers": len(tickers),
        "success_count": success_cnt,
        "cached_count": cached_cnt,
        "failed_count": fail_cnt,
        "total_rows": total_rows,
        "elapsed_seconds": round(elapsed, 2),
        "details": results,
    }
    logger.info(f"Completed acquisition: {success_cnt + cached_cnt}/{len(tickers)} available ({total_rows} rows in {elapsed:.1f}s).")
    return summary
