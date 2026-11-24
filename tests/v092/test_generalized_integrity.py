"""
Tests for StockReason v0.9.2 Generalized Corporate Actions & Segment Gap Integrity Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.canonical.integrity import (
    detect_unadjusted_corporate_actions_and_gaps,
    mask_generalized_unadjusted_target_windows,
)


def test_generalized_detection_catches_all_four_legacy_events():
    # Load panel data sample containing ADANIENT, TATAMOTORS, VEDL, CUPID
    panel_df = pd.read_parquet(
        "data/v0.9.1/features/panel_features.parquet",
        columns=["symbol", "date", "close", "adj_close", "target_return_21d"]
    )
    events_df = detect_unadjusted_corporate_actions_and_gaps(panel_df)
    
    # Verify legacy hardcoded 4 are detected automatically
    detected_pairs = set(zip(events_df["symbol"], events_df["date"]))
    
    assert ("ADANIENT", "2015-06-03") in detected_pairs
    assert ("TATAMOTORS", "2025-10-14") in detected_pairs
    assert ("VEDL", "2026-04-30") in detected_pairs
    assert ("CUPID", "2024-10-28") in detected_pairs

    # Verify that the total number of detected events generalizes beyond the 4
    assert len(events_df) > 50


def test_mask_generalized_unadjusted_target_windows():
    # Mini synthetic dataframe with an unadjusted demerger drop of 50%
    dates = pd.date_range("2023-01-01", periods=30, freq="B").strftime("%Y-%m-%d").tolist()
    closes = [100.0] * 15 + [50.0] * 15 # 50% drop at index 15
    
    df = pd.DataFrame({
        "symbol": ["TESTSYM"] * 30,
        "date": dates,
        "date_str": dates,
        "close": closes,
        "adj_close": closes,
        "target_return_5d": [0.0] * 30,
        "target_price_5d": closes
    })
    
    df_masked, events_df, counts = mask_generalized_unadjusted_target_windows(df, horizons=[5])
    assert len(events_df) >= 1
    assert counts[5] > 0
    # Window before index 15 should have target_return_5d set to NaN
    assert np.isnan(df_masked.loc[14, "target_return_5d"])
    assert np.isnan(df_masked.loc[15, "target_return_5d"])
