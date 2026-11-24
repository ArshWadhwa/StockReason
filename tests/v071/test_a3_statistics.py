"""
Unit tests for Workstream A3: Honest statistics, block bootstrap, and power analysis.
"""

import pytest
import numpy as np
import pandas as pd

from src.v071.honest_statistics import (
    block_bootstrap_ic,
    compute_non_overlapping_ic,
    compute_newey_west_tstat,
    adjust_multiple_testing,
    compute_power_analysis,
    evaluate_mechanical_verdict,
)


def test_bootstrap_recovers_planted_ic():
    # Planted IC of 0.05 across 200 dates
    np.random.seed(42)
    planted_mean = 0.05
    noise = np.random.normal(loc=planted_mean, scale=0.10, size=200)
    res = block_bootstrap_ic(noise, block_length=10, n_replications=2000, seed=42)

    assert abs(res["mean_ic"] - planted_mean) < 0.02
    assert res["ci_95"][0] <= planted_mean <= res["ci_95"][1]


def test_bootstrap_zero_ic_interval_includes_zero():
    # True IC = 0.0
    np.random.seed(101)
    zero_noise = np.random.normal(loc=0.0, scale=0.10, size=200)
    res = block_bootstrap_ic(zero_noise, block_length=10, n_replications=2000, seed=101)

    assert res["ci_95"][0] <= 0.0 <= res["ci_95"][1]
    verdict = evaluate_mechanical_verdict(res["ci_95"], "zero_baseline", "21D")
    assert "No demonstrated edge" in verdict


def test_non_overlapping_evaluation():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    ics = np.linspace(0.01, 0.10, 100)
    df = pd.DataFrame({"date": dates, "ic": ics})

    res = compute_non_overlapping_ic(df, horizon=21)
    assert res["n_offsets"] == 21
    assert len(res["offset_records"]) == 21
    assert 0.01 <= res["min_offset_ic"] <= res["max_offset_ic"] <= 0.10


def test_newey_west_hac():
    np.random.seed(42)
    ar1 = np.empty(300)
    ar1[0] = 0.05
    for t in range(1, 300):
        ar1[t] = 0.4 * ar1[t - 1] + np.random.normal(0, 0.1)

    mean_val, t_stat, p_val = compute_newey_west_tstat(ar1, max_lag=21)
    assert abs(mean_val) < 0.10
    assert not np.isnan(t_stat)
    assert 0.0 <= p_val <= 1.0


def test_multiple_testing_adjustment():
    raw_p = {"m1": 0.01, "m2": 0.04, "m3": 0.20, "m4": 0.80}
    adj_bonf = adjust_multiple_testing(raw_p, method="bonferroni")
    adj_bh = adjust_multiple_testing(raw_p, method="benjamini_hochberg")

    assert adj_bonf["m1"] == pytest.approx(0.04)
    assert adj_bonf["m4"] == 1.0
    assert adj_bh["m1"] <= adj_bonf["m1"]
    assert adj_bh["m2"] <= adj_bonf["m2"]


def test_power_analysis():
    # 50 independent dates (e.g. non-overlapping 21D over 4 years)
    res = compute_power_analysis(n_independent_dates=50, target_ic=0.03, std_ic=0.15)
    # Detectable IC at 80% power
    assert res["min_detectable_ic_at_80pct_power"] > 0.03
    assert res["is_sufficient_power"] is False
    assert res["n_dates_required_for_target_ic"] > 100
