"""
Tests for StockReason v0.9.2 Exact Per-Observation Target-End Timestamp Purging.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.validation.purged_cv import (
    compute_per_observation_target_end_dates,
    ExactPerObservationPurgedCV,
)


def test_missing_sessions_per_observation_purging():
    # Market calendar: 20 trading dates from 2023-01-02 to 2023-01-30
    all_dates = pd.date_range("2023-01-02", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    
    # Stock A trades continuously every day
    # Stock B is suspended/halted between session 5 and session 15 (skips 10 trading sessions)
    stock_a_dates = all_dates
    stock_b_dates = all_dates[:5] + all_dates[15:]
    
    df_a = pd.DataFrame({"symbol": "STOCK_A", "date_str": stock_a_dates, "close": 100.0})
    df_b = pd.DataFrame({"symbol": "STOCK_B", "date_str": stock_b_dates, "close": 200.0})
    df = pd.concat([df_a, df_b], ignore_index=True)
    
    # Horizon H = 5
    df_with_ends = compute_per_observation_target_end_dates(df, horizons=[5])
    
    # Check Stock A session 0: target end date is session 5 date
    a_row_0 = df_with_ends[(df_with_ends["symbol"] == "STOCK_A") & (df_with_ends["date_str"] == all_dates[0])].iloc[0]
    assert a_row_0["target_end_date_5d"] == all_dates[5]
    
    # Check Stock B session 4 (just before halt): target end date is its 5th future session, which is all_dates[19]!
    b_row_4 = df_with_ends[(df_with_ends["symbol"] == "STOCK_B") & (df_with_ends["date_str"] == all_dates[4])].iloc[0]
    assert b_row_4["target_end_date_5d"] == all_dates[19]
    
    # Now set validation start at all_dates[12]
    # Under fixed 5-day market shift, Stock B at all_dates[4] would seem to end at all_dates[9] (< all_dates[12])!
    # But its ACTUAL target realization is at all_dates[19] (> all_dates[12]), leaking 7 sessions!
    cv = ExactPerObservationPurgedCV(embargo_sessions=2)
    train_sessions = all_dates[:10]
    val_start = all_dates[12]
    
    purged_idx, audit = cv.purge_train_by_actual_target_end(
        df_with_ends,
        train_sessions=train_sessions,
        val_start_date=val_start,
        horizon=5,
        all_market_dates=all_dates,
    )
    
    # The audit must show NO leakage detected
    assert audit["leakage_detected"] is False
    assert audit["margin_calendar_days"] > 0
    
    # And Stock B's observation at all_dates[4] must have been PURGED
    purged_df = df_with_ends.loc[purged_idx]
    leaky_sample = purged_df[(purged_df["symbol"] == "STOCK_B") & (purged_df["date_str"] == all_dates[4])]
    assert len(leaky_sample) == 0, "Stock B observation spanning trading halt must be purged!"
