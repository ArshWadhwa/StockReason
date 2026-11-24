"""
Tests for StockReason v0.9.2 Multi-Horizon Conformal Calibration.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.calibration.intervals import MultiHorizonConformalCalibrator


def test_multi_horizon_conformal_calibrator():
    calibrator = MultiHorizonConformalCalibrator(horizons=[1, 5, 21, 63, 126])
    
    np.random.seed(42)
    n = 1000
    y_true = np.random.normal(0.02, 0.05, n)
    y_pred = np.random.normal(0.02, 0.02, n)
    vols = np.full(n, 0.25)
    
    # Fit across horizons
    for h in [1, 5, 21, 63, 126]:
        calibrator.fit_horizon(h, y_true, y_pred, vols)
        # Check quantiles are fitted
        assert 0.90 in calibrator.quantiles[h]
        assert calibrator.quantiles[h][0.90] > 0
        
    # Check predict_price_interval
    interval = calibrator.predict_price_interval(
        pred_return=0.03,
        stock_vol_annual=0.30,
        current_price=1000.0,
        horizon=21,
        coverage=0.95
    )
    assert interval["pred_price"] == 1030.0
    assert interval["lower_price_inr"] < 1030.0
    assert interval["upper_price_inr"] > 1030.0
    assert interval["margin_inr"] > 0
