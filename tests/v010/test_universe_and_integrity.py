"""
Unit tests for StockReason v0.10 Universe Definitions & Data Integrity.
"""

import pytest
import pandas as pd
import numpy as np

from src.v010.canonical.universe import audit_historical_universe_membership, compare_universe_training_regimes
from src.v010.canonical.integrity import resolve_engine_b_metric_contradiction, verify_target_price_consistency
from src.v091.features.pipeline import FEATURE_COLS_21


def test_universe_membership_and_controlled_comparison():
    # Load small subset of panel to verify execution
    panel_df = pd.read_parquet(
        "data/v0.9.1/features/panel_features.parquet",
        columns=["date", "symbol", "close", "target_return_21d"] + FEATURE_COLS_21
    )
    panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")
    
    # Audit membership
    audit = audit_historical_universe_membership(panel_df)
    assert "train_total_observations" in audit
    assert "train_top500_eligible" in audit
    assert audit["train_top500_eligible"] > 0
    assert audit["train_broad_only"] > 0

    # Compare regimes on a fast subsample of 5 stocks
    sample_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
    sub_panel = panel_df[panel_df["symbol"].isin(sample_symbols)].dropna(subset=FEATURE_COLS_21 + ["target_return_21d"]).copy()
    
    # Ensure some marked as Top-500
    sub_panel["is_top500_constituent"] = True
    
    regimes_df = compare_universe_training_regimes(sub_panel, features=FEATURE_COLS_21, target_col="target_return_21d")
    assert len(regimes_df) == 6 # 3 regimes x 2 models
    assert set(regimes_df["regime"].unique()) == {
        "1. Top-500 Only Training",
        "2. Broad Universe Training",
        "3. Broad Training + Top-500 Weighting",
    }
    assert all(regimes_df["return_mae"] > 0)


def test_metric_contradiction_resolution_and_price_consistency():
    # Synthetic evaluation dataframe
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for i in range(20):
            rows.append({
                "date_str": d,
                "pred_lgb": np.random.randn(),
                "pred_mom": np.random.randn() if (i % 5 != 0) else np.nan, # Some NaNs in momentum
                "target_demeaned_21d": np.random.randn(),
                "close": 100.0 + np.random.rand() * 50.0,
                "target_return_21d": 0.05 + np.random.randn() * 0.02,
            })
    df_eval = pd.DataFrame(rows)

    res = resolve_engine_b_metric_contradiction(df_eval)
    assert "mean_unpaired_lgb_ic" in res
    assert "mean_unpaired_mom_ic" in res
    assert "mean_paired_date_by_date_diff" in res
    assert res["n_paired_dates"] == 10

    # Target price consistency
    price_res = verify_target_price_consistency(df_eval, horizons=[21])
    assert price_res["21d"]["consistency_verified"] is True
    assert price_res["21d"]["has_negative_prices"] is False
