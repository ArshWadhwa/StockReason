import pandas as pd
import numpy as np
from src.m1_price_data.cleaner import clean_price_data

def test_clean_price_data_deduplication():
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "ticker": ["RELIANCE.NS", "RELIANCE.NS", "RELIANCE.NS"],
        "open": [2500.0, 2510.0, 2520.0],
        "high": [2550.0, 2560.0, 2570.0],
        "low": [2490.0, 2500.0, 2510.0],
        "close": [2540.0, 2550.0, 2560.0],
        "volume": [1000, 1100, 1200]
    }
    df = pd.DataFrame(data)
    cleaned = clean_price_data(df)
    assert len(cleaned) == 2, "Failed to remove duplicate date-ticker row"
    assert cleaned.iloc[0]["close"] == 2550.0, "Did not keep last duplicate row"

def test_clean_price_data_missing_values():
    data = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "ticker": ["TCS.NS", "TCS.NS", "TCS.NS"],
        "open": [3500.0, np.nan, 3520.0],
        "high": [3550.0, 3560.0, 3570.0],
        "low": [3490.0, 3500.0, 3510.0],
        "close": [3540.0, 3550.0, 3560.0],
        "volume": [1000, np.nan, 1200]
    }
    df = pd.DataFrame(data)
    cleaned = clean_price_data(df)
    assert not cleaned["open"].isna().any(), "Forward/backward fill failed for open price"
    assert cleaned.loc[cleaned["date"] == "2024-01-02", "open"].values[0] == 3500.0
