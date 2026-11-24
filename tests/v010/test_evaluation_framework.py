"""
Unit tests for StockReason v0.10 Baseline Suite & Walk-Forward Stability.
"""

import pytest
import numpy as np
import pandas as pd

from src.v010.evaluation.baselines import evaluate_ten_strict_baselines
from src.v010.evaluation.walk_forward import run_walk_forward_stability_breakdown
from src.v091.features.pipeline import FEATURE_COLS_21


def test_ten_strict_baselines_evaluation():
    np.random.seed(42)
    n_train = 300
    n_val = 100
    
    dates_tr = pd.date_range("2021-01-01", periods=n_train, freq="D").strftime("%Y-%m-%d")
    dates_val = pd.date_range("2023-01-01", periods=n_val, freq="D").strftime("%Y-%m-%d")

    def make_df(dates):
        data = {col: np.random.randn(len(dates)) for col in FEATURE_COLS_21}
        data["date_str"] = dates
        data["close"] = 100.0 + np.abs(np.random.randn(len(dates))) * 50.0
        data["target_return_21d"] = 0.02 + np.random.randn(len(dates)) * 0.05
        data["sector"] = np.random.choice(["IT", "BANK", "PHARMA"], size=len(dates))
        data["beta_60d"] = 1.0 + np.random.randn(len(dates)) * 0.2
        data["nifty_ret_21d"] = 0.01 + np.random.randn(len(dates)) * 0.03
        return pd.DataFrame(data)

    df_tr = make_df(dates_tr)
    df_val = make_df(dates_val)

    baselines_df = evaluate_ten_strict_baselines(df_tr, df_val, horizon=21, features=FEATURE_COLS_21)
    assert len(baselines_df) >= 10
    assert "1. No-Change (P_hat = P_t)" in baselines_df["baseline_name"].values
    assert "2. Historical Mean Return" in baselines_df["baseline_name"].values
    assert "6. Regularized Market-Aware CAPM" in baselines_df["baseline_name"].values
    assert "10. Existing v0.9.2 LightGBM Model" in baselines_df["baseline_name"].values
    assert all(baselines_df["return_mae"] > 0)

    # Test stability breakdown
    df_val["pred_lgb"] = 0.02 + np.random.randn(len(df_val)) * 0.03
    df_val["market_regime"] = np.random.choice(["Bullish_LowVol", "Bearish_Stress"], size=len(df_val))

    breakdown = run_walk_forward_stability_breakdown(df_val, "target_return_21d", "pred_lgb")
    assert "per_year" in breakdown
    assert "per_sector" in breakdown
    assert "per_regime" in breakdown
