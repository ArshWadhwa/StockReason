import pandas as pd
import numpy as np
from src.m1_price_data.lag_features import generate_lag_features

def test_no_future_data_leakage():
    dates = pd.date_range("2024-01-01", periods=10)
    df = pd.DataFrame({
        "date": dates,
        "ticker": "TCS.NS",
        "return_1d": [0.01 * i for i in range(10)],
        "volume": [1000 * i for i in range(10)]
    })

    lagged_df = generate_lag_features(df, lag_periods=[1, 2, 5])

    # For row index 0 (first date), lag_1 must be NaN because t-1 does not exist
    assert pd.isna(lagged_df.iloc[0]["return_1d_lag_1"]), "Row 0 lag_1 should be NaN"

    # For row index 1 (second date), lag_1 must equal return_1d at row 0
    assert lagged_df.iloc[1]["return_1d_lag_1"] == df.iloc[0]["return_1d"], "Lag_1 at index 1 must equal return_1d at index 0"

    # For row index 5, lag_5 must equal return_1d at row 0
    assert lagged_df.iloc[5]["return_1d_lag_5"] == df.iloc[0]["return_1d"], "Lag_5 at index 5 must equal return_1d at index 0"
