"""
Unit tests for Workstream A1: Leakage and sanity test battery.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.v07.common.lockbox import load_panel
from src.v07.features.feature_sets import get_feature_set
from src.v07.models.purged_cv import PurgedWalkForwardCV
from src.v071.a1_leakage_battery import (
    run_truncation_test,
    run_target_construction_audit,
    run_tuning_touch_audit,
    run_cross_sectional_normalization_audit,
    run_single_feature_ic_scan,
    run_timing_sanity_test,
    run_fold_boundary_audit,
)


@pytest.fixture(scope="module")
def panel():
    return load_panel(exclude_lockbox=True, exclude_embargo=True)


def test_a1_truncation_test(panel):
    res = run_truncation_test(panel, n_dates=10)
    assert res["status"] == "PASSED"
    assert res["mismatches_count"] == 0


def test_a1_target_construction(panel):
    res = run_target_construction_audit(panel, n_pairs=10)
    assert res["status"] == "PASSED"
    assert res["max_error"] < 1e-6


def test_a1_tuning_touch_audit():
    res = run_tuning_touch_audit()
    assert res["status"] == "PASSED"


def test_a1_cross_sectional_normalization(panel):
    res = run_cross_sectional_normalization_audit(panel)
    assert res["status"] == "PASSED"
    assert res["max_mean_deviation_from_zero"] < 1e-12


def test_a1_single_feature_ic_scan(panel):
    feats = [c for c in get_feature_set("C") if c in panel.columns]
    res = run_single_feature_ic_scan(panel, feats[:30])
    assert res["status"] in ["PASSED", "INVESTIGATE"]
    # Check that no feature has absurd IC > 0.50 (blatant leak)
    for r in res["top_20_features"]:
        assert r["abs_ic"] < 0.30


def test_a1_timing_sanity(panel):
    feats = [c for c in get_feature_set("C") if c in panel.columns][:10]
    res = run_timing_sanity_test(panel, feats)
    assert res["status"] == "PASSED"
    assert res["is_leaky_inflated"] is True


def test_a1_fold_boundary_purging(panel):
    res = run_fold_boundary_audit(panel, n_folds=4)
    assert res["all_folds_purged"] is True
