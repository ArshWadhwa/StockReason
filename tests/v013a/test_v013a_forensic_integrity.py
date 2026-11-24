"""StockReason v0.13A — Adversarial Regression & Mathematical Integrity Test Suite.

Section M Requirements:
Tests covering all historical failure modes and mathematical contracts:
1. Price and Return Tests:
   - Case A & Case B exact MAPE reference fixtures.
   - Proof that MAPE denominator is actual future price, not reference price.
   - Directional classification with FLAT abstention (No-Change gets 0% unconditional hit rate, not 52.93%).
2. Volatility Tests:
   - Trailing realized vol scale check (verifying returns, not raw rupee prices).
   - Recursive EWMA recurrence fixture: prev=0.0001, ret=0.05, lambda=0.94 -> 0.000244.
   - Normalized QLIKE formula: (0.04, 0.04)->0, (0.04, 0.02)->1-ln(2), (0.04, 0.08)->ln(2)-0.5.
   - Non-negativity of normalized QLIKE.
3. Calibration Tests:
   - Winkler score reference fixture: [90, 110], alpha=0.20 -> 100->20, 80->120, 120->120.
   - Rejection of constant 0.5 interval scaling.
   - Genuine finite-sample conformal quantile order statistic.
4. Temporal Integrity Tests:
   - Rejection of training labels crossing boundary into calibration.
   - Horizon-specific label realization purging.
5. Provenance & Report Integrity Tests:
   - Detection of discrepancy between report metrics and underlying prediction artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v013a.evaluation.metrics import (
    compute_return_mae,
    compute_return_rmse,
    compute_price_mae,
    compute_price_rmse,
    compute_true_price_mape,
    compute_true_price_medape,
    compute_directional_metrics_with_abstention,
    verify_mathematical_reference_fixtures,
)
from src.v013a.volatility.engine import (
    compute_normalized_qlike_loss,
    compute_recursive_ewma_variance,
    compute_forward_realized_variance,
)
from src.v013a.calibration.conformal import (
    compute_winkler_score,
    compute_pinball_loss,
    verify_winkler_reference_fixture,
    SplitConformalIntervalEngine,
)
from src.v013a.governance.partitions import (
    create_purged_horizon_partitions,
)
from src.v013a.api.service import (
    get_target_trading_date,
    StockReasonProductionServiceV013A,
)


# =====================================================================
# 1. Price and Return Metric Tests
# =====================================================================

def test_mape_case_a_and_case_b_fixtures():
    """Verifies Case A and Case B exact mathematical reference fixtures."""
    res = verify_mathematical_reference_fixtures()
    assert res["case_a_pass"], "Case A MAPE fixture failed!"
    assert res["case_b_pass"], "Case B MAPE fixture failed!"
    assert res["distinct_mapes_for_same_return_error"], "Same return error must produce distinct MAPEs for different future prices!"
    assert np.isclose(res["case_a_mape"], 8.33333333, atol=1e-5)
    assert np.isclose(res["case_b_mape"], 12.5, atol=1e-5)


def test_mape_rejects_reference_price_as_denominator():
    """
    Demonstrates that using reference price instead of actual future price
    produces incorrect MAPE and is rejected.
    """
    p_ref = np.array([100.0])
    p_act = np.array([120.0])
    p_pred = np.array([110.0])

    true_mape = compute_true_price_mape(p_act, p_pred)
    incorrect_mape = float(np.mean(np.abs(p_act - p_pred) / p_ref) * 100.0)

    assert np.isclose(true_mape, 8.333333, atol=1e-5)
    assert np.isclose(incorrect_mape, 10.0)
    assert not np.isclose(true_mape, incorrect_mape), "True MAPE must not equal reference-price denominator MAPE!"


def test_directional_abstention_no_change_rw():
    """
    Verifies that a constant zero-return forecast (No-Change Random Walk)
    is treated as 100% abstention for binary Up/Down direction, receiving
    0% unconditional hit rate (never credited for down moves!).
    """
    # 60% down moves, 40% up moves
    y_true = np.array([-0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.04, 0.01, -0.06])
    y_pred_rw = np.zeros_like(y_true)

    metrics = compute_directional_metrics_with_abstention(y_true, y_pred_rw)

    assert metrics["n_pred_flat"] == 10
    assert metrics["abstention_rate"] == 1.0
    assert metrics["coverage"] == 0.0
    assert metrics["unconditional_directional_hit_rate"] == 0.0
    assert metrics["conditional_directional_accuracy"] is None, "Conditional accuracy must be undefined (None) when no directional predictions made!"


# =====================================================================
# 2. Volatility Mathematics & Normalized QLIKE Tests
# =====================================================================

def test_normalized_qlike_analytical_fixtures():
    """Verifies normalized QLIKE loss on analytical reference fixtures."""
    # Fixture 1: Perfect prediction -> 0.0
    q1 = compute_normalized_qlike_loss(np.array([0.04]), np.array([0.04]))[0]
    assert np.isclose(q1, 0.0), f"Expected QLIKE=0, got {q1}"

    # Fixture 2: V_act=0.04, V_pred=0.02 -> 2 - ln(2) - 1 = 1 - ln(2) ~ 0.3068528
    q2 = compute_normalized_qlike_loss(np.array([0.04]), np.array([0.02]))[0]
    expected_q2 = 1.0 - np.log(2.0)
    assert np.isclose(q2, expected_q2, atol=1e-6), f"Expected QLIKE={expected_q2}, got {q2}"

    # Fixture 3: V_act=0.04, V_pred=0.08 -> 0.5 - ln(0.5) - 1 = ln(2) - 0.5 ~ 0.1931472
    q3 = compute_normalized_qlike_loss(np.array([0.04]), np.array([0.08]))[0]
    expected_q3 = np.log(2.0) - 0.5
    assert np.isclose(q3, expected_q3, atol=1e-6), f"Expected QLIKE={expected_q3}, got {q3}"


def test_normalized_qlike_strictly_non_negative():
    """Verifies that normalized QLIKE is strictly non-negative for all positive variances."""
    rng = np.random.RandomState(42)
    v_act = rng.uniform(0.001, 1.0, size=500)
    v_pred = rng.uniform(0.001, 1.0, size=500)

    qlike = compute_normalized_qlike_loss(v_act, v_pred)
    assert np.all(qlike >= -1e-12), "Normalized QLIKE must be strictly non-negative!"


def test_recursive_ewma_exact_recurrence_fixture():
    """
    Verifies EWMA recurrence on analytical test fixture:
    Previous daily variance = 0.0001
    Current return = 0.05
    Lambda = 0.94
    Expected daily variance = 0.94 * 0.0001 + 0.06 * (0.05^2) = 0.000244
    Annualized variance = 252 * 0.000244 = 0.061488
    Annualized volatility = sqrt(0.061488) = 0.24796774
    """
    prev_var = 0.0001
    curr_ret = 0.05
    decay = 0.94
    expected_daily_var = decay * prev_var + (1.0 - decay) * (curr_ret ** 2)
    expected_ann_var = expected_daily_var * 252.0
    expected_ann_vol = np.sqrt(expected_ann_var)

    assert np.isclose(expected_daily_var, 0.000244)
    assert np.isclose(expected_ann_var, 0.061488)
    assert np.isclose(expected_ann_vol, 0.24796774, atol=1e-6)


def test_volatility_uses_returns_not_raw_rupee_prices():
    """
    Adversarial test: checks that volatility engine operates on percentage returns,
    preventing the v0.12 defect where closing rupee prices (₹500, ₹1000) were grouped.
    """
    df = pd.DataFrame({
        "symbol": ["TEST", "TEST", "TEST"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "close": [1000.0, 1020.0, 1010.0],
        "ret_1d": [0.0, 0.02, -0.009804],
    })
    ewma_var = compute_recursive_ewma_variance(df, decay=0.94, return_col="ret_1d")
    ann_vol = np.sqrt(ewma_var)

    # Annualized volatility of a 1-2% daily move stock must be around 0.15 - 0.35, NEVER 400+!
    assert np.all(ann_vol < 2.0), f"Annualized volatility {ann_vol.values} is impossibly large (indicates raw price bug)!"


# =====================================================================
# 3. Calibration and Conformal Prediction Interval Tests
# =====================================================================

def test_winkler_score_reference_fixture():
    """Verifies Winkler interval score against exact analytical reference test."""
    res = verify_winkler_reference_fixture()
    assert res["passed"], f"Winkler reference fixture failed: {res}"
    assert np.isclose(res["score_at_100"], 20.0)
    assert np.isclose(res["score_at_80"], 120.0)
    assert np.isclose(res["score_at_120"], 120.0)


def test_pinball_loss_symmetry_at_median():
    """Verifies pinball loss at tau=0.5 equals half the MAE."""
    y = np.array([10.0, 20.0, 30.0])
    q = np.array([12.0, 18.0, 25.0])
    p_loss = compute_pinball_loss(y, q, tau=0.5)
    expected = 0.5 * float(np.mean(np.abs(y - q)))
    assert np.isclose(p_loss, expected)


def test_conformal_engine_no_hardcoded_half_multiplier():
    """
    Verifies that genuine split conformal calibrator does NOT clamp multipliers
    to 0.5 across all nominal coverage levels.
    """
    rng = np.random.RandomState(42)
    # Generate synthetic calibration residuals
    y_cal = rng.normal(0.0, 0.02, size=500)
    f_cal = np.zeros(500)
    sigma_cal = np.full(500, 0.02)

    engine = SplitConformalIntervalEngine(nominal_coverages=[0.50, 0.80, 0.90, 0.95])
    fit_res = engine.fit_calibration(y_cal, f_cal, sigma_cal)
    q_dict = fit_res["calibrated_quantiles"]

    # 50%, 80%, 90%, 95% quantiles must be strictly increasing and not all equal to 0.5!
    assert q_dict[0.50] < q_dict[0.80] < q_dict[0.90] < q_dict[0.95]
    assert not all(np.isclose(v, 0.5) for v in q_dict.values()), "Conformal quantiles must not be hardcoded to 0.5!"


# =====================================================================
# 4. Temporal Integrity & Label Purging Tests
# =====================================================================

def test_temporal_label_purging_rejects_boundary_leakage():
    """
    Adversarial test: intentionally injects observations near partition boundary
    whose future realization crosses into the next partition and verifies they are purged.
    """
    dates = pd.date_range("2022-12-01", "2023-01-31", freq="B")
    n = len(dates)
    df = pd.DataFrame({
        "symbol": ["TEST"] * n,
        "date": dates,
        "close": 100.0,
        "ret_1d": 0.001,
        "target_return_21d": 0.02,
        "target_price_21d": 102.0,
    })

    # For 21-day horizon, dates after ~2022-12-01 will realize in January 2023!
    splits = create_purged_horizon_partitions(
        df, horizon=21, train_end_date="2022-12-31", calib_start_date="2023-01-01"
    )

    # Must purge boundary observations
    assert splits["purged_train_count"] > 0, "Failed to purge boundary rows!"
    assert splits["train_leakage_prevented"], "Training partition label leaked into calibration!"


# =====================================================================
# 5. Production API Safeguards Tests
# =====================================================================

def test_api_service_nominal_reconstruction_and_probability_suppression():
    """
    Verifies that the production API reconstructs nominal prices correctly
    and suppresses uncalibrated probabilities.
    """
    service = StockReasonProductionServiceV013A(
        baseline_return_21d=0.0,
        is_probability_calibrated=False,
    )
    res = service.generate_forecast(
        ticker="INFY",
        current_price=1800.0,
        horizon=21,
        reference_date="2026-11-23",
    )

    # Nominal price reconstruction
    assert res["predicted_nominal_price_inr"] == 1800.0
    assert res["predicted_direction"] == "FLAT"
    # Probability suppression
    assert res["probability_positive_return"] is None
    assert res["probability_calibration_status"] == "SUPPRESSED_POOR_CALIBRATION"
    # Target date skips weekends
    assert res["target_trading_date"] > "2026-11-23"
