"""
Tests for StockReason v0.9.2 Paired Comparison & Bootstrap Difference CIs.
"""

import pytest
import pandas as pd
import numpy as np

from src.v092.models.comparison import (
    compute_paired_time_block_bootstrap_ci,
    compare_lgbm_vs_momentum_paired,
)


def test_paired_bootstrap_interval_properties():
    # Synthetic difference series with clear positive difference
    np.random.seed(42)
    diffs_positive = np.random.normal(loc=0.05, scale=0.01, size=100)
    m, low, high, is_sig = compute_paired_time_block_bootstrap_ci(diffs_positive, block_size=10)
    assert is_sig is True
    assert low > 0 and high > 0

    # Synthetic difference centered at 0
    diffs_null = np.random.normal(loc=0.0, scale=0.05, size=100)
    m_null, low_null, high_null, is_sig_null = compute_paired_time_block_bootstrap_ci(diffs_null, block_size=10)
    assert is_sig_null is False
    assert low_null < 0 and high_null > 0


def test_compare_lgbm_vs_momentum_paired():
    dates = [f"2023-01-{i:02d}" for i in range(1, 25)] * 10
    n = len(dates)
    df = pd.DataFrame({
        "date_str": dates,
        "pred_lgb": np.random.normal(0.01, 0.05, n),
        "pred_mom": np.random.normal(0.01, 0.05, n),
        "target_ret": np.random.normal(0.01, 0.05, n),
        "target_demeaned": np.random.normal(0.0, 0.05, n),
    })
    res = compare_lgbm_vs_momentum_paired(
        df,
        pred_lgb_col="pred_lgb",
        pred_mom_col="pred_mom",
        target_ret_col="target_ret",
        target_demeaned_col="target_demeaned",
    )
    assert "engine_a_return_mae_diff" in res
    assert "engine_b_rank_ic_diff" in res
    assert "ci_95" in res["engine_b_rank_ic_diff"]
