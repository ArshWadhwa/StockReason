"""
Unit tests for StockReason v0.8 Corporate Actions Auditor and Cleaner Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.v08.cleaner import clean_stock_dataframe, clean_universe_panel
from src.v08.corporate_actions import detect_corporate_actions_from_series, compute_consistent_adjusted_ohlcv
from src.v08.contracts import validate_ohlcv_dataframe


def test_split_detection_from_series():
    # Construct synthetic 2-for-1 split where unadjusted price halves but adj_close is continuous
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    # Day 0, 1, 2: unadj close = 2000; Day 3, 4, 5: unadj close = 1000 (after 2-for-1 split)
    close = [2000.0, 2010.0, 2020.0, 1015.0, 1020.0, 1025.0]
    # Adjusted close series retroactively adjusted (pre-split divided by 2)
    adj_close = [1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0]

    df = pd.DataFrame({
        "ticker": "SPLITCO.NS",
        "date": dates,
        "open": close,
        "high": [c + 10 for c in close],
        "low": [c - 10 for c in close],
        "close": close,
        "adj_close": adj_close,
        "volume": [10000] * 6,
    })

    events = detect_corporate_actions_from_series(df)
    assert len(events) == 1, f"Expected 1 corporate action event, found {len(events)}"
    ev = events[0]
    assert ev.action_type == "SPLIT"
    assert ev.split_ratio == 2.0
    assert ev.ex_date == dates[3].strftime("%Y-%m-%d")


def test_clean_stock_dataframe_anomalies_and_invariants():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    # Row 0: normal
    # Row 1: duplicate date of row 0
    # Row 2: non-positive price
    # Row 3: high < open (boundary violation)
    # Row 4: low > close (boundary violation)
    # Row 5: normal
    raw = pd.DataFrame({
        "ticker": "TEST.NS",
        "date": [dates[0], dates[0], dates[1], dates[2], dates[3], dates[4]],
        "open": [100.0, 100.5, -5.0, 110.0, 115.0, 120.0],
        "high": [105.0, 105.5, 10.0, 105.0, 125.0, 125.0],  # row 3 high (105) < open (110)
        "low": [95.0, 95.5, -10.0, 95.0, 120.0, 115.0],   # row 4 low (120) > close (118)
        "close": [102.0, 102.5, 5.0, 108.0, 118.0, 122.0],
        "adj_close": [102.0, 102.5, 5.0, 108.0, 118.0, 122.0],
        "volume": [1000, 1050, 2000, 3000, 4000, 5000],
    })

    cleaned, stats = clean_stock_dataframe(raw, "TEST.NS")

    # Invariant assertions
    assert stats["duplicate_rows_removed"] == 1
    assert stats["non_positive_rows_dropped"] == 1
    assert stats["ohlc_bounds_adjusted"] >= 2

    # Check contract validity
    is_valid, errors = validate_ohlcv_dataframe(cleaned, is_canonical=True)
    assert is_valid, f"Cleaned dataframe violated contract: {errors}"


def test_turnover_invariance_in_consistent_adjusted_ohlcv():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({
        "ticker": "ABC.NS",
        "date": dates,
        "open": [200.0, 200.0, 200.0],
        "high": [210.0, 210.0, 210.0],
        "low": [190.0, 190.0, 190.0],
        "close": [200.0, 200.0, 200.0],
        "adj_close": [100.0, 100.0, 100.0],  # adj_factor = 0.5
        "volume": [1000, 2000, 3000],
    })

    adj_df = compute_consistent_adjusted_ohlcv(df)
    assert np.allclose(adj_df["adj_open"], 100.0)
    assert np.allclose(adj_df["adj_high"], 105.0)
    assert np.allclose(adj_df["adj_low"], 95.0)
    # Volume should double when price halves to preserve turnover
    assert np.allclose(adj_df["adj_volume"], [2000, 4000, 6000])


def test_clean_universe_panel():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    p1 = pd.DataFrame({
        "ticker": "STOCK1.NS",
        "date": dates,
        "open": [100.0, 102.0, 104.0],
        "high": [105.0, 107.0, 109.0],
        "low": [98.0, 100.0, 102.0],
        "close": [101.0, 103.0, 105.0],
        "adj_close": [101.0, 103.0, 105.0],
        "volume": [1000, 1200, 1400],
    })
    p2 = pd.DataFrame({
        "ticker": "STOCK2.NS",
        "date": dates,
        "open": [50.0, 52.0, 54.0],
        "high": [55.0, 57.0, 59.0],
        "low": [48.0, 50.0, 52.0],
        "close": [51.0, 53.0, 55.0],
        "adj_close": [51.0, 53.0, 55.0],
        "volume": [5000, 5200, 5400],
    })
    panel = pd.concat([p1, p2], ignore_index=True)

    cleaned_panel, stats = clean_universe_panel(panel)
    assert stats["total_tickers"] == 2
    assert stats["output_rows"] == 6
    is_valid, errors = validate_ohlcv_dataframe(cleaned_panel, is_canonical=True)
    assert is_valid, f"Canonical panel failed validation: {errors}"
