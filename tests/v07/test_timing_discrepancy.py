"""
test_timing_discrepancy.py
--------------------------
Demonstrates the timing discrepancy between the legacy training sequence builder
(src.m3_modeling.data_loader) and the live inference pipeline (src.m3_modeling.predict).

DISCREPANCY EXPLANATION:
------------------------
In legacy training (data_loader.py):
  1. `Target_1D` is computed as `close.shift(-1) / close - 1`, so row i has target `close[i+1] / close[i] - 1`.
  2. The sequence loop uses:
       all_x.append(features[i - sequence_length : i, :])
       all_y.append(targets[i, :])
       all_dates.append(dates[i])
  3. `features[i - sequence_length : i]` contains rows [i - sequence_length, ..., i - 1].
     The feature window ENDS at row i - 1!
  4. But `targets[i]` is the return from row i to row i + 1 (`close[i+1] / close[i] - 1`),
     and `dates[i]` is recorded as the sample date.
  5. Consequently:
     - The return from row i - 1 to row i is completely skipped.
     - The features ending at row i - 1 are asked to predict the return starting from row i,
       OR row i's features are omitted even though row i is the origin of the return.

In live inference (predict.py):
  1. At date T (latest row), `window = scaled[-SEQUENCE_LENGTH:]` takes features ending at row T (inclusive).
  2. The model prediction is applied directly to `current_price = latest_row['close']` (row T).
  3. Therefore, live inference assumes the input window ends at the forecast origin T,
     whereas legacy training omitted row T from the input window (1-day gap).
"""

import numpy as np
import pandas as pd
import pytest

from src.m3_modeling.data_loader import _build_sequences_per_ticker


def test_timing_discrepancy_on_synthetic_data():
    """
    Construct a synthetic series with 5 days where price and feature = day index.
    Verify that legacy builder pairs features ending at day i-1 with targets starting at day i.
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    # Day 0: 100, Day 1: 110, Day 2: 121, Day 3: 133.1, Day 4: 146.41, Day 5: 161.051
    # 10% daily return
    prices = [100.0 * (1.1 ** i) for i in range(6)]
    
    # Feature is simply the day index (0, 1, 2, 3, 4, 5)
    df = pd.DataFrame({
        "ticker": ["TEST.NS"] * 6,
        "date": dates,
        "close": prices,
        "feature_day": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })

    # Compute target as in data_loader.py
    df["Target_1D"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df["Target_5D"] = 0.0
    df["Target_21D"] = 0.0
    df["Target_126D"] = 0.0
    
    # Dropna: last row has NaN target, so df has 5 rows (indices 0, 1, 2, 3, 4)
    df = df.dropna().reset_index(drop=True)
    assert len(df) == 5

    seq_len = 2
    feature_cols = ["feature_day"]
    
    x, y, sample_dates = _build_sequences_per_ticker(df, feature_cols, sequence_length=seq_len)

    # In the loop: range(seq_len, len(features)) -> i = 2, 3, 4
    assert len(x) == 3

    # For the first sequence (i = 2):
    # sample_dates[0] is df['date'][2] -> Day 2 ('2024-01-03')
    # Expected target y[0][0] is Target_1D[2] = return from Day 2 to Day 3 = 10% (0.10)
    assert sample_dates[0] == pd.Timestamp("2024-01-03")
    assert pytest.approx(y[0][0], rel=1e-4) == 0.10

    # NOW CHECK THE INPUT FEATURES x[0]:
    # x[0] was formed by features[i-seq_len:i] = features[0:2] = [feature_day 0, feature_day 1]
    input_feature_days = x[0].flatten().tolist()
    assert input_feature_days == [0.0, 1.0]

    # DEMONSTRATION OF THE DISCREPANCY:
    # The last feature in the input window is Day 1 (index 1), NOT Day 2 (index 2)!
    # Meanwhile, the sample date is recorded as Day 2, and the return is measured from Day 2 to Day 3.
    # Day 1 -> Day 2 return is missing, and Day 2's feature is NEVER in the input window!
    last_feature_day = input_feature_days[-1]
    sample_day_index = 2.0  # Day 2 corresponds to sample_dates[0]

    assert last_feature_day == 1.0
    assert sample_day_index == 2.0
    # There is a 1-day timing lag: last_feature_day != sample_day_index
    assert last_feature_day < sample_day_index, (
        f"Timing discrepancy confirmed: input features end at day {last_feature_day}, "
        f"while sample date and target origin are at day {sample_day_index}."
    )
