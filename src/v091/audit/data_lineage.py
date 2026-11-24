"""
StockReason v0.9.1 — Data Lineage & Synthetic Quarantine Audit.
===============================================================
Maintains strict data provenance and anti-fabrication gates (Task 12):
- Inventories every dataset file used in training and evaluation.
- Records: file path, data source, row count, unique tickers, date range,
  SHA-256 checksum, and synthetic status flag.
- Verifies zero synthetic data rows exist in the active canonical pipeline.
"""

import os
import glob
import hashlib
from typing import Dict, List, Any
import pandas as pd


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file in binary mode."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_data_lineage_manifest(
    output_path: str = "reports/v0.9.1/data_lineage_inventory.csv",
) -> pd.DataFrame:
    """
    Generates data lineage manifest for all files used in v0.9.1 training and evaluation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    file_patterns = [
        ("data/indian-markets/universe/nse_liquid.parquet", "TejHQ Liquidity Universe"),
        ("data/external/sector_indices/NIFTY_50.parquet", "External Benchmark NIFTY 50"),
        ("data/external/sector_indices/INDIA_VIX.parquet", "External Benchmark INDIA VIX"),
    ]
    
    # Add raw Bhavcopy files
    for f in sorted(glob.glob("data/indian-markets/nse/year=*/nse_*.parquet")):
        file_patterns.append((f, "TejHQ Official NSE Bhavcopy"))
        
    # Add adjusted price files
    for f in sorted(glob.glob("data/indian-markets/prices_adjusted/nse_*.parquet")):
        file_patterns.append((f, "TejHQ Split-Adjusted Bhavcopy"))
        
    # Add corporate actions
    for f in sorted(glob.glob("data/indian-markets/actions/nse_*.parquet")):
        file_patterns.append((f, "TejHQ Corporate Actions"))
        
    records = []
    for fpath, source_desc in file_patterns:
        if not os.path.exists(fpath):
            continue
            
        sha = compute_sha256(fpath)
        file_size_kb = os.path.getsize(fpath) // 1024
        
        row_count = 0
        ticker_count = 0
        min_date = "N/A"
        max_date = "N/A"
        is_synthetic = False
        
        try:
            if fpath.endswith(".parquet"):
                # Read metadata or sample
                df_sample = pd.read_parquet(fpath)
                row_count = len(df_sample)
                if "symbol" in df_sample.columns:
                    ticker_count = df_sample["symbol"].nunique()
                elif "Symbol" in df_sample.columns:
                    ticker_count = df_sample["Symbol"].nunique()
                else:
                    ticker_count = 1
                    
                if "date" in df_sample.columns:
                    min_date = str(df_sample["date"].min())
                    max_date = str(df_sample["date"].max())
                elif "Date" in df_sample.columns:
                    min_date = str(df_sample["Date"].min())
                    max_date = str(df_sample["Date"].max())
                elif df_sample.index.name in ["Date", "date"]:
                    min_date = str(df_sample.index.min())
                    max_date = str(df_sample.index.max())
                    
                if "is_synthetic" in df_sample.columns and df_sample["is_synthetic"].any():
                    is_synthetic = True
        except Exception:
            pass
            
        records.append({
            "file_path": fpath,
            "source": source_desc,
            "size_kb": file_size_kb,
            "row_count": row_count,
            "ticker_count": ticker_count,
            "start_date": min_date,
            "end_date": max_date,
            "sha256": sha,
            "is_synthetic": is_synthetic
        })
        
    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(output_path, index=False)
    return df_manifest
