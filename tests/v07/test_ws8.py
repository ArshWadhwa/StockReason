"""Tests for StockReason v0.7 Uncertainty Package (WS8).

Validates:
1. Directional probability derivation: P(excess > 0), P(net > 0).
2. Calibration: Isotonic (N >= 1000) and Platt (N < 1000), Brier score improvement, ECE reduction.
3. Conformal prediction: 80% nominal coverage achieved within +/- 3% on synthetic Gaussian data.
4. Conformalized Quantile Regression (CQR) interval generation and coverage.
5. Strict causality and no future look-ahead in rolling-window conformal calibration.
6. Adaptive Conformal Inference (ACI) dynamic adaptation under distribution shift.
7. MC-dropout interval empirical undercoverage comparison.
8. Schema serialization, field validation, and backward-compatible deprecation of confidence.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from src.v07.uncertainty.conformal import (
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    SplitConformalRegressor,
    compare_intervals_with_mc_dropout,
    compute_conformal_quantile,
    compute_coverage,
    compute_interval_metrics,
    compute_winkler_score,
    evaluate_coverage_by_regime,
    evaluate_coverage_over_time,
    run_rolling_conformal_simulation,
)
from src.v07.uncertainty.direction import (
    DirectionalCalibrator,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    compute_reliability_curve,
    derive_directional_probabilities,
    generate_reliability_diagram,
)
from src.v07.uncertainty.schema import (
    DeprecatedConfidenceWarning,
    StockUncertaintyForecast,
    UncertaintyHorizonOutput,
    build_uncertainty_horizon,
    upgrade_legacy_horizon_output,
)


class TestDirectionalProbabilities:
    """Tests for directional probability derivation and calibration."""

    def test_derive_directional_probabilities_scalar_and_array(self):
        # Scalar
        res = derive_directional_probabilities(
            predicted_return=0.02,
            uncertainty_std=0.02,
            benchmark_return=0.01,
            cost_bps=15.0,  # 0.0015
        )
        assert 0.0 <= res["prob_up"] <= 1.0
        assert 0.0 <= res["prob_excess_up"] <= 1.0
        assert 0.0 <= res["prob_net_up"] <= 1.0
        assert 0.0 <= res["prob_excess_net_up"] <= 1.0

        # With positive return and lower benchmark/cost:
        # P(return > 0) > P(return > cost) > P(return - bench > cost)
        assert res["prob_up"] > res["prob_net_up"]
        assert res["prob_net_up"] > res["prob_excess_net_up"]

        # Vector / array
        rets = np.array([-0.02, 0.0, 0.02])
        stds = np.array([0.02, 0.02, 0.02])
        res_arr = derive_directional_probabilities(rets, uncertainty_std=stds)
        assert isinstance(res_arr["prob_up"], np.ndarray)
        assert len(res_arr["prob_up"]) == 3
        assert res_arr["prob_up"][0] < 0.5 < res_arr["prob_up"][2]
        assert abs(res_arr["prob_up"][1] - 0.5) < 1e-4

    def test_isotonic_calibration_improves_brier_score(self):
        """
        Verify that isotonic calibration on uncalibrated probabilities
        improves Brier score and reduces ECE on test holdout data.
        """
        np.random.seed(42)
        n = 2000
        # Generate miscalibrated probabilities (systematically distorted/overconfident)
        raw_x = np.random.uniform(-2, 2, n)
        true_prob = 1.0 / (1.0 + np.exp(-raw_x))
        # Uncalibrated score is distorted (overconfident at extremes)
        uncal_prob = np.where(true_prob > 0.5, true_prob ** 0.4, true_prob ** 2.5)
        uncal_prob = np.clip(uncal_prob, 0.01, 0.99)
        y = (np.random.uniform(0, 1, n) < true_prob).astype(float)

        # Split 50/50 OOF calibration vs evaluation
        oof_idx = slice(0, 1000)
        test_idx = slice(1000, 2000)

        cal = DirectionalCalibrator(method="auto")
        cal.fit(uncal_prob[oof_idx], y[oof_idx])
        assert cal.method_used_ == "isotonic"
        assert cal.n_samples_ == 1000

        # Predict calibrated on test holdout
        test_uncal = uncal_prob[test_idx]
        test_cal = cal.predict_proba(test_uncal)
        test_y = y[test_idx]

        brier_uncal = compute_brier_score(test_y, test_uncal)
        brier_cal = compute_brier_score(test_y, test_cal)
        assert brier_cal < brier_uncal, f"Expected brier_cal ({brier_cal}) < brier_uncal ({brier_uncal})"

        ece_uncal = compute_ece(test_y, test_uncal)["ece"]
        ece_cal = compute_ece(test_y, test_cal)["ece"]
        assert ece_cal < ece_uncal, f"Expected ece_cal ({ece_cal}) < ece_uncal ({ece_uncal})"

    def test_platt_scaling_selected_for_small_samples(self):
        """Verify Platt scaling is chosen automatically when N < 1000."""
        np.random.seed(123)
        n = 400
        uncal = np.random.uniform(0.1, 0.9, n)
        y = (uncal > 0.5).astype(float)

        cal = DirectionalCalibrator(method="auto")
        cal.fit(uncal, y)
        assert cal.method_used_ == "platt"
        assert cal.n_samples_ == 400

        preds = cal.predict_proba(uncal[:10])
        assert len(preds) == 10
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_metrics_and_reliability_diagram_generation(self, tmp_path):
        """Verify metric functions and diagram plotting without error."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.65, 0.4, 0.85, 0.75])

        brier = compute_brier_score(y_true, y_prob)
        loss = compute_log_loss(y_true, y_prob)
        ece_dict = compute_ece(y_true, y_prob, n_bins=5)

        assert brier > 0.0
        assert loss > 0.0
        assert 0.0 <= ece_dict["ece"] <= 1.0

        plot_file = tmp_path / "test_calibration.png"
        res = generate_reliability_diagram(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=5,
            title="Test Diagram",
            save_path=plot_file,
            y_prob_uncalibrated=y_prob * 0.9,
        )
        assert plot_file.exists()
        assert "calibrated" in res
        assert "uncalibrated" in res


class TestConformalPrediction:
    """Tests for Conformal Prediction intervals."""

    def test_conformal_80_percent_coverage_synthetic_gaussian(self):
        """
        Synthetic data with known Gaussian noise:
        80% conformal interval achieves ~80% coverage (+/- 3%).
        """
        np.random.seed(42)
        n_samples = 4000
        true_std = 0.025

        # True process: y = f(x) + epsilon, epsilon ~ N(0, true_std^2)
        x = np.random.uniform(-1, 1, n_samples)
        f_x = 0.04 * x
        noise = np.random.normal(0, true_std, n_samples)
        y = f_x + noise
        y_pred = f_x  # unbiased model prediction

        # Split 50/50 calibration and test
        n_cal = 2000
        y_cal, y_hat_cal = y[:n_cal], y_pred[:n_cal]
        y_test, y_hat_test = y[n_cal:], y_pred[n_cal:]

        sc = SplitConformalRegressor(alpha=0.20)
        sc.fit(y_true_cal=y_cal, y_pred_cal=y_hat_cal)

        lower, upper = sc.predict_interval(y_hat_test)
        coverage = compute_coverage(y_test, lower, upper)

        # Nominal 80% coverage, expected 80% +/- 3% (0.77 to 0.83)
        assert 0.77 <= coverage <= 0.83, (
            f"Conformal coverage {coverage:.4f} outside nominal 80% +/- 3% bounds"
        )

        metrics = compute_interval_metrics(y_test, lower, upper, alpha=0.20)
        assert abs(metrics["coverage_gap"]) <= 0.03
        assert metrics["mean_width"] > 0.0

    def test_cqr_adaptive_coverage(self):
        """
        Verify Conformalized Quantile Regression produces valid 80% intervals.
        """
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 3)
        # Heteroscedastic noise: std depends on X[:, 0]
        hetero_std = 0.01 + 0.03 * np.abs(X[:, 0])
        noise = np.random.normal(0, hetero_std)
        y = 0.02 * X[:, 0] + noise

        # Train base models on first 800, calibrate on next 600, test on remaining 600
        cqr = ConformalizedQuantileRegression(alpha=0.20, n_estimators=50, max_depth=3)
        cqr.fit_base_models(X[:800], y[:800])
        cqr.calibrate(y[800:1400], X_cal=X[800:1400])

        lower, upper = cqr.predict_interval(X_test=X[1400:])
        cov = compute_coverage(y[1400:], lower, upper)
        # Should be within +/- 5% for heteroscedastic data of moderate sample size (nominal 80%)
        assert 0.75 <= cov <= 0.85, f"CQR coverage {cov:.4f} outside expected tolerance"
        # Verify interval widths vary (adaptive)
        widths = upper - lower
        assert np.std(widths) > 0.001

    def test_no_future_lookahead_in_rolling_calibration(self):
        """
        Strict verification of no future look-ahead in rolling calibration:
        Modifying future ground truth values at t > k MUST NOT change
        the conformal interval predicted at t = k.
        """
        np.random.seed(99)
        n = 300
        window = 60
        y_true_orig = np.random.normal(0, 0.02, n)
        y_pred = np.zeros(n)

        # Baseline run
        sim_orig = run_rolling_conformal_simulation(
            y_true=y_true_orig,
            y_pred=y_pred,
            window_size=window,
            alpha=0.20,
            adaptive_aci=False,
        )

        # Modify only future values after index 120 (e.g. inject extreme crisis shock)
        y_true_shocked = y_true_orig.copy()
        shock_idx = 120
        y_true_shocked[shock_idx:] = y_true_shocked[shock_idx:] * 100.0 + 5.0

        sim_shocked = run_rolling_conformal_simulation(
            y_true=y_true_shocked,
            y_pred=y_pred,
            window_size=window,
            alpha=0.20,
            adaptive_aci=False,
        )

        # All intervals prior to shock_idx (eval indices 0 to shock_idx - window) MUST be bitwise identical!
        pre_shock_slice = slice(0, shock_idx - window)
        np.testing.assert_array_almost_equal(
            sim_orig["lowers"][pre_shock_slice],
            sim_shocked["lowers"][pre_shock_slice],
            decimal=10,
            err_msg="Lookahead detected! Pre-shock intervals were affected by future modifications.",
        )
        np.testing.assert_array_almost_equal(
            sim_orig["uppers"][pre_shock_slice],
            sim_shocked["uppers"][pre_shock_slice],
            decimal=10,
            err_msg="Lookahead detected! Pre-shock upper bounds were affected by future modifications.",
        )

    def test_adaptive_conformal_inference_dynamic_adaptation(self):
        """
        Verify ACI dynamically widens intervals after sudden volatility shock.
        """
        np.random.seed(77)
        # First 100 days calm (std=0.01), next 50 days violent shock (std=0.08)
        calm = np.random.normal(0, 0.01, 100)
        shock = np.random.normal(0, 0.08, 50)
        y = np.concatenate([calm, shock])
        pred = np.zeros_like(y)

        aci = AdaptiveConformalInference(target_alpha=0.20, gamma=0.10, window_size=40)
        aci.reset(initial_residuals=np.abs(calm[:40]))

        widths = []
        for t in range(40, len(y)):
            l, u = aci.predict_interval(pred[t])
            widths.append(u - l)
            aci.step(y_true=y[t], lower=l, upper=u, y_pred=pred[t])

        # Width during violent shock should expand significantly compared to calm period
        calm_width = np.mean(widths[:40])
        shock_width = np.mean(widths[60:])
        assert shock_width > calm_width * 2.0, (
            f"Expected shock width ({shock_width:.4f}) to expand relative to calm ({calm_width:.4f})"
        )

    def test_evaluate_coverage_over_time_and_regime(self):
        """Verify time-series and regime diagnostic breakdowns."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        y = np.random.normal(0, 0.02, 100)
        lower = y - 0.03
        upper = y + 0.03
        regimes = np.array(["BULL"] * 50 + ["BEAR"] * 50)

        reg_res = evaluate_coverage_by_regime(y, lower, upper, regimes, alpha=0.20)
        assert "BULL" in reg_res and "BEAR" in reg_res
        assert reg_res["BULL"]["empirical_coverage"] == 1.0

        time_res = evaluate_coverage_over_time(dates, y, lower, upper, rolling_window=20, alpha=0.20)
        assert len(time_res["rolling_coverage"]) == 81
        assert time_res["nominal_coverage"] == 0.80

    def test_mc_dropout_comparison_exposes_undercoverage(self):
        """
        Evaluate on the same test predictions:
        MC-dropout interval (mean +/- 1.96*std) has severe undercoverage,
        while conformal interval achieves nominal coverage.
        """
        np.random.seed(42)
        n = 1000
        actual_noise = np.random.normal(0, 0.03, n)
        y_true = actual_noise
        y_pred = np.zeros(n)

        # MC-dropout standard deviation represents only epistemic model uncertainty (~0.004)
        mc_std = np.full(n, 0.004)

        # Conformal interval calibrated on first 500 samples
        sc = SplitConformalRegressor(alpha=0.20)
        sc.fit(y_true[:500], y_pred[:500])
        l_conf, u_conf = sc.predict_interval(y_pred[500:])

        res = compare_intervals_with_mc_dropout(
            y_true=y_true[500:],
            point_pred=y_pred[500:],
            mc_std=mc_std[500:],
            lower_conformal=l_conf,
            upper_conformal=u_conf,
            nominal_conformal=0.80,
            nominal_mc=0.95,
        )

        mc_cov = res["mc_dropout"]["empirical_coverage"]
        conf_cov = res["conformal"]["empirical_coverage"]

        # MC coverage with 0.004 std on 0.03 noise is ~10-25%, far below nominal 95%
        assert mc_cov < 0.35, f"MC coverage should be severely degraded, got {mc_cov:.4f}"
        assert abs(conf_cov - 0.80) <= 0.04, f"Conformal coverage should be ~80%, got {conf_cov:.4f}"
        assert res["mc_undercoverage_deficit"] > 0.50


class TestSchemaAndSerialization:
    """Tests for Pydantic uncertainty schemas and deprecation behaviors."""

    def test_valid_uncertainty_horizon_output(self):
        horizon = build_uncertainty_horizon(
            expected_return=0.015,
            current_price=1000.0,
            prob_up=0.68,
            interval_80=(980.0, 1030.0),
            forecast_origin_date="2026-09-08",
            prob_up_calibrated=True,
            prob_excess_up=0.58,
            prob_net_up=0.65,
            interval_coverage_observed=0.804,
            n_calibration_samples=500,
            model_version="0.7.0",
        )
        assert horizon.expected_return == 0.015
        assert horizon.predicted_price == 1015.0
        assert horizon.prob_up == 0.68
        assert horizon.prob_up_calibrated is True
        assert horizon.interval_80 == (980.0, 1030.0)
        assert horizon.lower_bound == 980.0
        assert horizon.upper_bound == 1030.0
        # Deprecated confidence should match prob_up when not passed
        assert horizon.confidence == 0.68

    def test_confidence_deprecation_warning(self):
        """Verify warning when legacy confidence is explicitly provided."""
        with pytest.warns(DeprecatedConfidenceWarning, match="confidence.*deprecated"):
            UncertaintyHorizonOutput(
                expected_return=0.01,
                predicted_price=101.0,
                prob_up=0.60,
                interval_80=(95.0, 105.0),
                forecast_origin_date="2026-09-08",
                confidence=0.85,  # Explicit deprecated field
            )

    def test_upgrade_legacy_horizon_output(self):
        """Verify seamless upgrading of legacy v0.6 horizon dictionary."""
        legacy = {
            "expected_return": 0.008677,
            "predicted_price": 1074.24,
            "confidence": 0.9619,
            "uncertainty_std": 0.003959,
            "lower_bound": 1065.98,
            "upper_bound": 1082.51,
        }
        upgraded = upgrade_legacy_horizon_output(
            legacy_data=legacy,
            current_price=1065.0,
            forecast_origin_date="2026-09-07",
            calibration_q=0.025,
            prob_up_calibrated=True,
            n_calibration_samples=400,
            interval_coverage_observed=0.801,
        )
        assert upgraded.prob_up_calibrated is True
        assert upgraded.n_calibration_samples == 400
        assert upgraded.interval_coverage_observed == 0.801
        # Conformal interval computed from current_price * (1 + ret +/- q)
        assert upgraded.interval_80[0] < upgraded.interval_80[1]
        assert upgraded.lower_bound == upgraded.interval_80[0]

    def test_full_stock_forecast_validation(self):
        """Verify complete ticker forecast structure."""
        h1d = build_uncertainty_horizon(
            expected_return=0.001,
            current_price=500.0,
            prob_up=0.55,
            interval_80=(492.0, 508.0),
            forecast_origin_date="2026-09-08",
        )
        forecast = StockUncertaintyForecast(
            ticker="TCS.NS",
            prediction_date="2026-09-08",
            current_price=500.0,
            horizons={"1D": h1d},
        )
        d = forecast.model_dump()
        assert d["ticker"] == "TCS.NS"
        assert d["horizons"]["1D"]["prob_up"] == 0.55
        assert d["horizons"]["1D"]["interval_80"] == (492.0, 508.0)
