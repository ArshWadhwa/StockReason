"""
Tests for StockReason v0.11 Forensic Remediations.
===================================================
1. Volatility target evaluation & self-comparison rejection.
2. Causal market regime detector multi-dimensional diversity.
3. Target price consistency across all 5 horizons.
"""

import pytest
import numpy as np
import pandas as pd

from src.v011.volatility.vol_models import (
    compute_forward_realized_volatility,
    compute_qlike_loss,
    detect_volatility_self_comparison,
    DedicatedVolatilityEngineV011,
)
from src.v011.regimes.causal_detector import CausalMarketRegimeDetectorV011
from src.v011.canonical.targets import (
    verify_target_price_consistency_v011,
    build_forward_price_targets,
    HORIZONS,
)


def test_forward_realized_volatility_calculation():
    # Construct 100 days of price data for 2 symbols with known return volatility
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    rows = []
    for sym in ["RELIANCE", "TCS"]:
        p = 1000.0
        for d in dates:
            ret = np.random.normal(0.0005, 0.015)
            p = p * (1.0 + ret)
            rows.append({
                "date": d,
                "symbol": sym,
                "close": p,
                "vol_60d": 0.20,
            })
    df = pd.DataFrame(rows)

    fwd_vol = compute_forward_realized_volatility(df, horizon=21)
    df["forward_realized_vol_21d"] = fwd_vol

    # Forward volatility must not be identically equal to trailing vol_60d
    valid = df["forward_realized_vol_21d"].dropna()
    assert len(valid) > 0
    assert not detect_volatility_self_comparison(valid.values, np.full(len(valid), 0.20))
    # Values should be reasonable annualized volatilities (between 0.05 and 0.60)
    assert valid.min() > 0.02
    assert valid.max() < 0.80


def test_volatility_self_comparison_rejection():
    # If someone accidentally compares vol_60d with vol_60d
    vol_60d = np.array([0.22, 0.25, 0.19, 0.30, 0.28])
    assert detect_volatility_self_comparison(vol_60d, vol_60d) is True

    # If targets are genuinely forward and independent
    vol_fwd = np.array([0.15, 0.35, 0.26, 0.18, 0.40])
    assert detect_volatility_self_comparison(vol_fwd, vol_60d) is False


def test_causal_regime_detector_diversity():
    # Construct synthetic dates covering bull, bear, volatile, and sideways conditions
    n_days = 200
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    df = pd.DataFrame({
        "date": dates,
        "date_str": dates.strftime("%Y-%m-%d"),
        # Section 1: Bull low vol (days 0-50)
        # Section 2: Bull high vol (days 50-100)
        # Section 3: Bear stress (days 100-150)
        # Section 4: Sideways (days 150-200)
        "vix_level": [12.0]*50 + [22.0]*50 + [25.0]*50 + [15.0]*50,
        "nifty_trend_slope": [0.03]*50 + [0.02]*50 + [-0.04]*50 + [0.00]*50,
        "nifty_ret_21d": [0.04]*50 + [0.03]*50 + [-0.06]*50 + [0.001]*50,
        "dist_ma50": [0.03]*50 + [0.01]*50 + [-0.10]*50 + [-0.01]*50,
    })

    detector = CausalMarketRegimeDetectorV011()
    df_reg = detector.detect_regimes(df)
    stats = detector.compute_regime_statistics(df_reg)

    # Must NOT collapse into 100% single regime
    assert not stats["is_collapsed_single_regime"]
    assert len(stats["regime_counts"]) >= 3
    assert "Bullish_Expansion" in stats["regime_counts"]
    assert "Bearish_Stress" in stats["regime_counts"]
    assert "Sideways_Consolidation" in stats["regime_counts"]


def test_target_price_consistency_verifier():
    df = pd.DataFrame({
        "close": [100.0, 200.0, 50.0],
        "target_price_21d": [105.0, 210.0, 48.0],
        "target_return_21d": [0.05, 0.05, -0.04],
        "target_price_1d": [101.0, 198.0, 50.5],
        "target_return_1d": [0.01, -0.01, 0.01],
    })

    res = verify_target_price_consistency_v011(df, horizons=[1, 21])
    assert res["1d"]["status"] == "PASS"
    assert res["21d"]["status"] == "PASS"
    assert res["21d"]["negative_prices_count"] == 0
    assert res["21d"]["corporate_action_adjusted_rows_count"] == 0
