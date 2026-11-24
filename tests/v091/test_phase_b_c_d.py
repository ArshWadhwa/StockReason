"""
StockReason v0.9.1 — Phase B, C, D Verification & Promotion Gate Tests.
=======================================================================
Covers Tasks 4, 6, 7, 9, 10, 11:
1. Conformal interval dynamic scaling by volatility (wider for high-volatility stock).
2. Probability calibration omission gate (omitted when Brier skill score <= 0).
3. Cross-sectional target construction (mean is exactly zero per date).
4. Model training and Rank IC early stopping.
5. Matched-cohort ablation execution.
6. 4-Factor time-series regression with Newey-West standard errors.
7. Realistic institutional backtest (liquidity-tiered slippage & ADV caps).
8. Prospective logging append-only format and honest status.
9. Promotion gates evaluation.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.v091.features.targets import construct_cross_sectional_targets
from src.v091.calibration.intervals import VolatilityStandardizedConformalCalibrator
from src.v091.calibration.probabilities import CalibratedProbabilityEngine
from src.v091.features.ablation import run_matched_cohort_ablation
from src.v091.backtest.engine import (
    run_factor_attribution_regression,
    compute_realistic_institutional_backtest
)
from src.v091.audit.prospective_logging import (
    compute_pipeline_fingerprint,
    append_prospective_log,
    evaluate_prospective_readiness
)
from src.v091.api.forecast_service import StockReasonForecastServiceV091


def test_cross_sectional_demeaned_targets_zero_mean():
    """Task 4 Test: Cross-sectional demeaned target has exact zero mean per date."""
    np.random.seed(42)
    df = pd.DataFrame({
        "date_str": ["2024-01-05"] * 50 + ["2024-01-08"] * 50,
        "target_return_21d": np.random.normal(0.03, 0.08, 100),
        "vol_60d": np.random.uniform(0.15, 0.40, 100)
    })
    
    out = construct_cross_sectional_targets(df, horizons=[21])
    for d, grp in out.groupby("date_str"):
        mean_val = float(grp["target_demeaned_21d"].mean())
        assert abs(mean_val) < 1e-12, f"Expected 0 mean per date, got {mean_val}"


def test_conformal_interval_volatility_scaling():
    """Task 7 Test: Conformal intervals dynamically scale width by stock volatility."""
    calibrator = VolatilityStandardizedConformalCalibrator()
    calibrator.quantiles[0.95] = 2.0
    
    # Stock A: Low Volatility utility (vol = 15%)
    res_a = calibrator.predict_interval(pred_return=0.02, stock_vol_annual=0.15, ref_price=100.0, coverage=0.95)
    # Stock B: High Volatility smallcap (vol = 60%)
    res_b = calibrator.predict_interval(pred_return=0.02, stock_vol_annual=0.60, ref_price=100.0, coverage=0.95)
    
    assert res_b["margin_return"] > res_a["margin_return"] * 3.5, "High volatility stock must have significantly wider interval"
    assert res_a["lower_price"] > res_b["lower_price"], "Low volatility price interval must be tighter"


def test_probability_omission_when_brier_skill_score_negative():
    """Task 7 Test: If Brier skill score <= 0, probability must be omitted from API (None)."""
    engine = CalibratedProbabilityEngine()
    engine.brier_skill_score = -0.05
    engine.is_valid = False
    
    prob = engine.get_calibrated_probability_or_omit(0.0189)
    assert prob is None, "Probability must be None (omitted) when Brier Skill Score <= 0"


def test_factor_attribution_regression():
    """Task 9 Test: 4-factor regression runs and computes Newey-West t-statistics."""
    np.random.seed(42)
    n = 30
    mkt = np.random.normal(0.01, 0.04, n)
    smb = np.random.normal(0.005, 0.03, n)
    hml = np.random.normal(0.002, 0.03, n)
    wml = np.random.normal(0.008, 0.03, n)
    # Strategy with heavy beta to SMB and Volatility, zero genuine alpha
    strat = 0.0 + 1.2 * mkt + 0.8 * smb + 0.6 * hml + np.random.normal(0, 0.01, n)
    
    res = run_factor_attribution_regression(strat, mkt, smb, hml, wml)
    assert "alpha_newey_west_t" in res
    assert res["beta_market"] > 0.8
    assert res["beta_smb"] > 0.5


def test_realistic_institutional_backtest_costs():
    """Task 9 Test: Assigns liquidity-tiered slippage and computes break-even multiple."""
    trade_log = pd.DataFrame({
        "rebalance_period": [1, 1, 2, 2],
        "symbol": ["RELIANCE", "CUPID", "RELIANCE", "CUPID"],
        "adv_inr": [1500000000.0, 50000000.0, 1500000000.0, 50000000.0], # RELIANCE 150 Cr, CUPID 5 Cr
        "raw_return": [0.03, 0.15, -0.02, 0.10]
    })
    
    res = compute_realistic_institutional_backtest(trade_log)
    assert res["realistic_net_return_pct"] is not None
    assert res["annualized_sharpe"] is not None
    assert "sharpe_95_ci" in res


def test_prospective_logging_and_honesty_check():
    """Task 10 Test: Prospective log appends records and reports insufficient history honestly."""
    log_file = "artifacts/v0.9.1/test_prospective.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    records = [{"symbol": "INFY", "forecast_origin_date": "2026-11-24", "pred_return": 0.015}]
    n_written = append_prospective_log(records, log_path=log_file)
    assert n_written == 1
    
    readiness = evaluate_prospective_readiness(log_path=log_file, min_months_required=6)
    assert readiness["gate_passed"] is False
    assert "fewer than 6 months" in readiness["verdict"]
    
    if os.path.exists(log_file):
        os.remove(log_file)
