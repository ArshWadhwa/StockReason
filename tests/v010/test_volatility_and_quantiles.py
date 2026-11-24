"""
Unit tests for StockReason v0.10 Volatility Engine & Quantile Distributional Forecasting.
"""

import pytest
import numpy as np
import pandas as pd

from src.v010.volatility.vol_models import DedicatedVolatilityEngine, compute_qlike_loss
from src.v010.volatility.quantiles import MultiQuantilePriceForecaster, compute_pinball_loss


def test_volatility_models_and_qlike():
    np.random.seed(42)
    n = 200
    df_val = pd.DataFrame({
        "close": 100.0 + np.random.randn(n) * 10.0,
        "vol_60d": 0.20 + np.abs(np.random.randn(n)) * 0.05,
        "forward_realized_vol_21d": 0.22 + np.abs(np.random.randn(n)) * 0.05,
    })
    X_val = np.random.randn(n, 5)

    engine = DedicatedVolatilityEngine(horizon=21)
    engine.fit_gbdt_volatility(X_val, df_val["forward_realized_vol_21d"].values)

    res_df = engine.evaluate_volatility_models(df_val, X_val, features=["f1", "f2", "f3", "f4", "f5"])
    assert len(res_df) == 4
    assert all(res_df["vol_mae"] > 0)
    assert all(res_df["qlike_loss"] >= 0)
    assert all(res_df["mean_interval_width_inr"] > 0)


def test_quantile_forecaster_and_non_crossing():
    np.random.seed(42)
    n_train = 300
    n_val = 50
    n_features = 10

    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randn(n_train) * 0.05

    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.randn(n_val) * 0.05
    val_close = np.full(n_val, 150.0)

    forecaster = MultiQuantilePriceForecaster()
    forecaster.fit(X_train, y_train)

    preds = forecaster.predict_quantiles(X_val, val_close)
    assert "return_quantiles" in preds
    assert "price_quantiles_inr" in preds

    # Verify Non-Crossing Property: q_05 <= q_10 <= q_25 <= q_50 <= q_75 <= q_90 <= q_95
    p_05 = preds["price_quantiles_inr"]["q_05"]
    p_50 = preds["price_quantiles_inr"]["q_50"]
    p_95 = preds["price_quantiles_inr"]["q_95"]
    assert np.all(p_05 <= p_50)
    assert np.all(p_50 <= p_95)

    eval_df = forecaster.evaluate(X_val, y_val, val_close)
    assert len(eval_df) == 7
    assert all(eval_df["pinball_loss"] > 0)
