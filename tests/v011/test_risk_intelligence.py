"""
Tests for StockReason v0.11 Risk Intelligence.
==============================================
1. Quantile non-crossing verification.
2. Positive price bound guarantee.
3. Downside risk and Brier score calibration.
"""

import pytest
import numpy as np
import pandas as pd

from src.v011.volatility.quantiles import (
    MultiQuantilePriceForecasterV011,
    compute_pinball_loss,
    compute_interval_score,
)
from src.v011.volatility.downside_risk import DownsideRiskEngineV011


def test_quantile_non_crossing_and_positive_prices():
    np.random.seed(42)
    n = 200
    X_tr = np.random.randn(n, 5)
    y_tr = np.random.normal(0.01, 0.08, n)

    forecaster = MultiQuantilePriceForecasterV011()
    forecaster.fit(X_tr, y_tr)

    X_val = np.random.randn(50, 5)
    current_prices = np.random.uniform(50, 2000, 50)

    preds = forecaster.predict_quantiles(X_val, current_prices, apply_conformal=False)

    ret_q = preds["return_quantiles"]
    price_q = preds["price_quantiles_inr"]

    # Test non-crossing for each row: q_05 <= q_10 <= q_25 <= q_50 <= q_75 <= q_90 <= q_95
    q_keys = ["q_05", "q_10", "q_25", "q_50", "q_75", "q_90", "q_95"]
    for i in range(len(q_keys) - 1):
        assert (ret_q[q_keys[i]] <= ret_q[q_keys[i + 1]]).all()
        assert (price_q[q_keys[i]] <= price_q[q_keys[i + 1]]).all()

    # Test non-negative prices
    for qk in q_keys:
        assert (price_q[qk] > 0).all()


def test_pinball_loss_and_interval_score():
    y_true = np.array([0.05, -0.02, 0.01])
    y_pred = np.array([0.03, -0.01, 0.02])
    pl = compute_pinball_loss(y_true, y_pred, 0.50)
    assert pl > 0.0

    lower = np.array([-0.05, -0.05, -0.05])
    upper = np.array([0.05, 0.05, 0.05])
    score = compute_interval_score(y_true, lower, upper, alpha=0.10)
    assert score > 0.0


def test_downside_risk_engine():
    np.random.seed(42)
    n = 200
    X_tr = np.random.randn(n, 4)
    y_tr = np.random.normal(0.02, 0.06, n)

    engine = DownsideRiskEngineV011()
    engine.fit(X_tr, y_tr)

    X_val = np.random.randn(50, 4)
    y_val = np.random.normal(0.02, 0.06, 50)
    dummy_quantiles = {"q_05": np.full(50, -0.10), "q_10": np.full(50, -0.07), "q_25": np.full(50, -0.03)}

    res = engine.evaluate_calibration(X_val, y_val, dummy_quantiles)
    assert "brier_score_model" in res
    assert "brier_skill_score" in res
    assert 0.0 <= res["brier_score_model"] <= 1.0
