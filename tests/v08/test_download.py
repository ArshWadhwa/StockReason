"""
Unit and integration tests for StockReason v0.8 Resilient OHLCV Downloader.
"""

import os
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.v08.download import (
    sanitize_ticker,
    unsanitize_ticker,
    standardize_ohlcv_dataframe,
    fetch_ticker_ohlcv,
    download_universe_ohlcv,
    DEFAULT_START_DATE,
)
from src.v08.contracts import validate_ohlcv_dataframe


@pytest.fixture
def temp_download_env(tmp_path):
    """Sets up isolated temporary directories for downloader tests."""
    raw_dir = tmp_path / "raw" / "prices"
    raw_dir.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True)
    checkpoint = meta_dir / "checkpoint.json"
    provenance = meta_dir / "provenance.json"
    return {
        "raw_dir": str(raw_dir),
        "checkpoint": str(checkpoint),
        "provenance": str(provenance),
    }


def test_sanitize_and_unsanitize():
    cases = [
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("^NSEI", "INDEX_NSEI"),
        ("^NSEBANK", "INDEX_NSEBANK"),
        ("USDINR=X", "USDINR_EQ_X"),
    ]
    for orig, sanitized in cases:
        s = sanitize_ticker(orig)
        assert s == sanitized, f"Expected {sanitized}, got {s}"
        assert unsanitize_ticker(s) == orig


def test_standardize_ohlcv_dataframe():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    mock_df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "Adj Close": [101.5, 102.5, 103.5, 104.5, 105.5],
        "Volume": [1000, 2000, 3000, 4000, 5000],
    })

    res = standardize_ohlcv_dataframe(mock_df, "TCS.NS")
    is_valid, errors = validate_ohlcv_dataframe(res)
    assert is_valid, f"Standardization produced invalid dataframe: {errors}"
    assert len(res) == 5
    assert res["ticker"].iloc[0] == "TCS.NS"
    assert res["date"].dtype == "datetime64[ns]"


def test_fetch_ticker_ohlcv_mocked_success(temp_download_env):
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    mock_history = pd.DataFrame({
        "Date": dates,
        "Open": np.linspace(100, 110, 10),
        "High": np.linspace(105, 115, 10),
        "Low": np.linspace(95, 105, 10),
        "Close": np.linspace(102, 112, 10),
        "Adj Close": np.linspace(101, 111, 10),
        "Volume": np.full(10, 50000),
    })

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_history
        mock_ticker_cls.return_value = mock_instance

        res = fetch_ticker_ohlcv(
            ticker="INFY.NS",
            start_date="2024-01-01",
            end_date="2024-01-15",
            output_dir=temp_download_env["raw_dir"],
            checkpoint_path=temp_download_env["checkpoint"],
            provenance_path=temp_download_env["provenance"],
        )

        assert res["status"] == "success"
        assert res["rows"] == 10
        assert os.path.exists(res["filepath"])

        # Check caching behavior on second call
        res_cached = fetch_ticker_ohlcv(
            ticker="INFY.NS",
            start_date="2024-01-01",
            end_date="2024-01-15",
            output_dir=temp_download_env["raw_dir"],
            checkpoint_path=temp_download_env["checkpoint"],
            provenance_path=temp_download_env["provenance"],
            incremental=True,
        )
        assert res_cached["status"] == "cached"
        assert res_cached["rows"] == 10


def test_fetch_ticker_ohlcv_failure_handling(temp_download_env):
    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        # Return empty dataframe on every attempt
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        res = fetch_ticker_ohlcv(
            ticker="NONEXISTENT.NS",
            start_date="2024-01-01",
            output_dir=temp_download_env["raw_dir"],
            checkpoint_path=temp_download_env["checkpoint"],
            provenance_path=temp_download_env["provenance"],
            max_retries=1,
            rate_limit_delay=0.01,
            v07_fallback_dir="nonexistent_dir",
        )

        assert res["status"] == "failed"
        assert res["rows"] == 0
        assert res["filepath"] is None
        assert res["error"] is not None
