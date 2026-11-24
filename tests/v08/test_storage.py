"""
Unit tests for StockReason v0.8 Partitioned Parquet Storage Engine and Panel Loader.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.v08.storage import (
    init_storage_directories,
    optimize_dtypes,
    write_parquet_atomic,
    compute_directory_manifest,
    STORAGE_LAYOUT,
)
from src.v08.panel_loader import (
    load_panel_subset,
    stream_ticker_chunks,
    stream_time_windows,
)


@pytest.fixture
def temp_storage_env(tmp_path):
    """Sets up temporary storage environment for testing."""
    base_dir = str(tmp_path / "data_v08")
    prices_dir = str(tmp_path / "data_v08" / "canonical" / "prices")
    os.makedirs(prices_dir, exist_ok=True)

    # Generate synthetic parquet files for 3 tickers
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    dates = pd.date_range("2023-01-01", periods=10, freq="B")

    for t in tickers:
        df = pd.DataFrame({
            "ticker": t,
            "date": dates,
            "open": np.linspace(100, 110, 10, dtype=np.float64),
            "high": np.linspace(105, 115, 10, dtype=np.float64),
            "low": np.linspace(95, 105, 10, dtype=np.float64),
            "close": np.linspace(102, 112, 10, dtype=np.float64),
            "adj_close": np.linspace(101, 111, 10, dtype=np.float64),
            "volume": np.full(10, 50000, dtype=np.int64),
        })
        out_f = os.path.join(prices_dir, f"{t}.parquet")
        write_parquet_atomic(df, out_f, optimize=True)

    return {
        "base_dir": base_dir,
        "prices_dir": prices_dir,
        "tickers": tickers,
    }


def test_optimize_dtypes():
    df = pd.DataFrame({
        "ticker": ["RELIANCE.NS", "TCS.NS"],
        "date": ["2024-01-01", "2024-01-02"],
        "price": [1500.50, 1520.75],
        "volume": [100000, 120000],
    })
    opt = optimize_dtypes(df)
    assert opt["price"].dtype == np.float32
    assert opt["ticker"].dtype.name == "category"
    assert opt["date"].dtype == "datetime64[ns]"


def test_atomic_write_and_manifest(temp_storage_env):
    prices_dir = temp_storage_env["prices_dir"]
    manifest_path = os.path.join(temp_storage_env["base_dir"], "manifest.json")

    manifest = compute_directory_manifest(prices_dir, output_manifest_path=manifest_path)
    assert manifest["total_files"] == 3
    assert manifest["total_rows"] == 30
    assert os.path.exists(manifest_path)
    assert len(manifest["files"]) == 3
    for f_info in manifest["files"]:
        assert len(f_info["sha256"]) == 64  # valid SHA-256 length


def test_load_panel_subset_filtering(temp_storage_env):
    prices_dir = temp_storage_env["prices_dir"]

    # 1. Load single ticker with projected columns
    df_single = load_panel_subset(
        tickers=["TCS.NS"],
        columns=["ticker", "date", "close"],
        canonical_dir=prices_dir,
    )
    assert len(df_single) == 10
    assert list(df_single.columns) == ["ticker", "date", "close"]
    assert (df_single["ticker"] == "TCS.NS").all()

    # 2. Date filtering
    df_dated = load_panel_subset(
        start_date="2023-01-05",
        end_date="2023-01-10",
        canonical_dir=prices_dir,
    )
    assert len(df_dated) > 0
    assert df_dated["date"].min() >= pd.to_datetime("2023-01-05")
    assert df_dated["date"].max() <= pd.to_datetime("2023-01-10")


def test_stream_ticker_chunks(temp_storage_env):
    prices_dir = temp_storage_env["prices_dir"]
    tickers = temp_storage_env["tickers"]

    chunks = list(stream_ticker_chunks(tickers, chunk_size=2, canonical_dir=prices_dir))
    assert len(chunks) == 2  # 3 tickers in chunks of 2 -> 2 chunks (2, 1)
    assert len(chunks[0]["ticker"].unique()) == 2
    assert len(chunks[1]["ticker"].unique()) == 1
