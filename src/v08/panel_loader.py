"""
StockReason v0.8 Memory-Efficient Panel Loader & Streaming Engine.
================================================================
Provides lazy loading, column pruning, row filtering, and chunked streaming
for multi-decade equity panels on commodity hardware.
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any, Generator
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.v08.download import sanitize_ticker, unsanitize_ticker
from src.v08.storage import optimize_dtypes

logger = logging.getLogger("v08.panel_loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_CANONICAL_PRICES_DIR = "data/v0.8/canonical/prices"


def load_panel_subset(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    canonical_dir: str = DEFAULT_CANONICAL_PRICES_DIR,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """
    Loads a subset of the canonical equity panel with column projection and date pruning.
    Avoids loading unused columns or unrequested tickers into memory.
    """
    if not os.path.exists(canonical_dir):
        logger.warning(f"Canonical directory {canonical_dir} does not exist.")
        return pd.DataFrame()

    # Determine files to load
    if tickers is not None and len(tickers) > 0:
        target_files = []
        for t in tickers:
            sanitized = sanitize_ticker(t)
            fpath = os.path.join(canonical_dir, f"{sanitized}.parquet")
            if os.path.exists(fpath):
                target_files.append(fpath)
            else:
                logger.debug(f"File for ticker {t} not found at {fpath}")
    else:
        target_files = sorted(glob.glob(os.path.join(canonical_dir, "*.parquet")))

    if not target_files:
        logger.warning(f"No parquet files found in {canonical_dir} matching requested criteria.")
        return pd.DataFrame()

    # Ensure required filtering columns are loaded even if not in projected list
    read_columns = list(columns) if columns is not None else None
    if read_columns is not None:
        if "date" not in read_columns:
            read_columns.append("date")
        if "ticker" not in read_columns:
            read_columns.append("ticker")

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    dfs = []
    for fpath in target_files:
        try:
            # PyArrow table read with projected columns
            table = pq.read_table(fpath, columns=read_columns)
            df = table.to_pandas()
            if df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

            # Row filtering on date
            if start_dt is not None:
                df = df[df["date"] >= start_dt]
            if end_dt is not None:
                df = df[df["date"] <= end_dt]

            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Sort deterministically
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # If original caller did not ask for date or ticker, prune them if desired
    if columns is not None:
        combined = combined[[c for c in columns if c in combined.columns]]

    if optimize_memory:
        combined = optimize_dtypes(combined)

    return combined


def stream_ticker_chunks(
    tickers: List[str],
    chunk_size: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    canonical_dir: str = DEFAULT_CANONICAL_PRICES_DIR,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator yielding panel DataFrames in chunks of N tickers.
    Enables commodity hardware with low RAM to process 500 stocks sequentially.
    """
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i : i + chunk_size]
        logger.debug(f"Streaming ticker chunk {i // chunk_size + 1} ({len(chunk_tickers)} tickers)...")
        df_chunk = load_panel_subset(
            tickers=chunk_tickers,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            canonical_dir=canonical_dir,
            optimize_memory=True,
        )
        if not df_chunk.empty:
            yield df_chunk


def stream_time_windows(
    tickers: Optional[List[str]] = None,
    start_date: str = "2011-11-01",
    end_date: Optional[str] = None,
    window_days: int = 252,
    columns: Optional[List[str]] = None,
    canonical_dir: str = DEFAULT_CANONICAL_PRICES_DIR,
) -> Generator[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame], None, None]:
    """
    Generator yielding panel DataFrames in rolling time windows (e.g. 1-year windows).
    """
    cur_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date) if end_date else pd.to_datetime(datetime.now(timezone.utc).date())

    while cur_start < final_end:
        cur_end = cur_start + pd.Timedelta(days=window_days)
        df_win = load_panel_subset(
            tickers=tickers,
            start_date=cur_start.strftime("%Y-%m-%d"),
            end_date=cur_end.strftime("%Y-%m-%d"),
            columns=columns,
            canonical_dir=canonical_dir,
            optimize_memory=True,
        )
        if not df_win.empty:
            yield cur_start, cur_end, df_win
        cur_start = cur_end + pd.Timedelta(days=1)
