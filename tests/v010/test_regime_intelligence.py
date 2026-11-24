"""
Unit tests for StockReason v0.10 Causal Regime Detection & Adaptive Forecasting.
"""

import pytest
import pandas as pd
import numpy as np

from src.v010.regimes.causal_detector import CausalMarketRegimeDetector
from src.v010.regimes.adaptive_ensemble import RegimeAdaptiveEnsemble


def test_causal_regime_detection_and_adaptive_ensemble():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=20, freq="B").strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for i in range(10):
            rows.append({
                "date_str": d,
                "symbol": f"STOCK_{i}",
                "close": 100.0 + np.random.randn() * 10.0,
                "vix_level": 14.0 if d < "2023-01-15" else 22.0,
                "nifty_trend_slope": 0.05 if d < "2023-01-15" else -0.05,
                "nifty_ret_21d": 0.03 if d < "2023-01-15" else -0.04,
                "target_ret_21d": 0.02 + np.random.randn() * 0.05,
            })
    df_sample = pd.DataFrame(rows)

    detector = CausalMarketRegimeDetector()
    df_regimes = detector.detect_regimes(df_sample)
    assert "market_regime" in df_regimes.columns
    stats = detector.compute_regime_statistics(df_regimes)
    assert "regime_frequencies_pct" in stats
    assert stats["total_trading_days"] == 20

    # Test Adaptive Ensemble
    n = len(df_sample)
    pred_mean = np.full(n, 0.02)
    pred_hyb = np.full(n, 0.025)
    pred_lgb = np.full(n, 0.03)
    target_ret = df_sample["target_ret_21d"].values

    ensemble = RegimeAdaptiveEnsemble()
    preds = ensemble.predict(df_regimes, pred_mean, pred_hyb, pred_lgb)
    assert len(preds["adaptive_return"]) == n
    assert len(preds["adaptive_price"]) == n
    assert all(preds["adaptive_price"] > 0)
    assert all(preds["fixed_control_price"] > 0)

    eval_df = ensemble.evaluate_comparison(df_regimes, target_ret, pred_mean, pred_hyb, pred_lgb)
    assert len(eval_df) == 5
    assert "1. Regime-Adaptive Ensemble" in eval_df["model_variant"].values
    assert "2. Fixed-Weight Control (Uniform)" in eval_df["model_variant"].values
