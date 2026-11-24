"""
Tests for StockReason v0.11 Stability, Drift & Promotion.
=========================================================
1. Drift monitor KS-test sensitivity.
2. Champion-challenger promotion gating logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.v011.stability.drift_monitor import ModelDriftMonitorV011
from src.v011.stability.champion_challenger import evaluate_champion_challenger_promotion, ModelLifecycleState


def test_drift_monitor_data_drift_detection():
    np.random.seed(42)
    # Reference data: mean 0, std 1
    ref_df = pd.DataFrame({
        "f1": np.random.normal(0, 1, 500),
        "f2": np.random.normal(0, 1, 500),
    })

    monitor = ModelDriftMonitorV011(ref_df, feature_cols=["f1", "f2"])

    # Incoming healthy data: same distribution
    inc_healthy = pd.DataFrame({
        "f1": np.random.normal(0, 1, 500),
        "f2": np.random.normal(0, 1, 500),
    })
    res_h = monitor.check_data_drift(inc_healthy)
    assert res_h["drift_status"] == "HEALTHY"

    # Incoming drifted data: mean shifted by 5 std devs
    inc_drifted = pd.DataFrame({
        "f1": np.random.normal(5, 1, 500),
        "f2": np.random.normal(5, 1, 500),
    })
    res_d = monitor.check_data_drift(inc_drifted)
    assert res_d["drift_status"] == "CRITICAL"
    assert res_d["drifted_features_count"] == 2


def test_champion_challenger_promotion_gates():
    # Scenario A: Challenger is worse than champion
    res_fail = evaluate_champion_challenger_promotion(
        champion_name="Historical_Mean",
        champion_mae=0.0895,
        challenger_name="Deep_Transformer",
        challenger_mae=0.0920,
        ci_low=0.0010,
        ci_high=0.0035,
        is_statistically_significant=True,
    )
    assert res_fail["final_state"] == ModelLifecycleState.REJECTED.value
    assert "REJECTED" in res_fail["verdict"]

    # Scenario B: Challenger is better historically but pending prospective validation
    res_exp = evaluate_champion_challenger_promotion(
        champion_name="Historical_Mean",
        champion_mae=0.0895,
        challenger_name="Residual_LightGBM",
        challenger_mae=0.0885,
        ci_low=-0.0015,
        ci_high=-0.0005,
        is_statistically_significant=True,
        has_prospective_pass=False,
    )
    assert res_exp["final_state"] == ModelLifecycleState.HISTORICALLY_VALIDATED.value

    # Scenario C: Challenger passes all 5 gates including prospective evidence
    res_prom = evaluate_champion_challenger_promotion(
        champion_name="Historical_Mean",
        champion_mae=0.0895,
        challenger_name="Residual_LightGBM",
        challenger_mae=0.0870, # > 2% improvement
        ci_low=-0.0030,
        ci_high=-0.0020,
        is_statistically_significant=True,
        has_prospective_pass=True,
    )
    assert res_prom["final_state"] == ModelLifecycleState.PRODUCTION_APPROVED.value
