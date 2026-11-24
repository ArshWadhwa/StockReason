"""
Unit tests for StockReason v0.8 Scaling Experiment Suite.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import pytest
import pandas as pd
import numpy as np

from src.v08.scaling_experiments import (
    block_bootstrap_rank_ic,
    simulate_topk_strategy,
    run_scaling_benchmark_experiment,
)
from src.v08.features import compute_technical_features, compute_multi_horizon_targets


def test_block_bootstrap_rank_ic():
    # Synthetic series centered around 0.05
    np.random.seed(42)
    ic_series = np.random.normal(loc=0.05, scale=0.02, size=100)
    lower_ci, upper_ci, p_val = block_bootstrap_rank_ic(ic_series, n_replications=500, block_length=5)

    assert lower_ci < upper_ci
    assert lower_ci > 0.0, f"Expected lower CI > 0, got {lower_ci}"
    assert p_val < 0.05


def test_simulate_topk_strategy():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    records = []
    for d in dates:
        for i in range(15):
            records.append({
                "date": d,
                "ticker": f"STOCK_{i}.NS",
                "pred": float(i),  # stock 14 always has highest prediction
                "target": 0.01 if i >= 10 else -0.01,  # top stocks always yield positive return
            })
    df = pd.DataFrame(records)
    metrics = simulate_topk_strategy(df, k=5, holding_days=21, cost_bps=20.0)

    assert "net_return_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["net_return_pct"] > 0.0


def test_scaling_benchmark_experiment_mock():
    # Build synthetic panel across 100 tickers and 300 days
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=250, freq="B")
    records = []
    tickers = [f"MOCK_{i}.NS" for i in range(25)]

    for t in tickers:
        base_p = 100.0 + np.random.rand() * 50.0
        p = base_p + np.cumsum(np.random.randn(len(dates)))
        p = np.maximum(p, 10.0)
        for dt_idx, dt in enumerate(dates):
            cp = p[dt_idx]
            records.append({
                "ticker": t,
                "date": dt,
                "open": cp,
                "high": cp * 1.02,
                "low": cp * 0.98,
                "close": cp,
                "adj_close": cp,
                "volume": 100000,
            })
    raw_panel = pd.DataFrame(records)
    feat_panel = compute_technical_features(raw_panel)
    panel_with_targets = compute_multi_horizon_targets(feat_panel, horizons=[21])

    # Run scaling experiment on mock panel
    results = run_scaling_benchmark_experiment(
        panel_df=panel_with_targets,
        common_test_tickers=tickers,
        test_start_date=dates[200].strftime("%Y-%m-%d"),
        horizon=21,
    )

    assert isinstance(results, dict)
    assert len(results) > 0
    first_cfg = list(results.values())[0]
    assert "exact_price_metrics" in first_cfg
    assert "stock_ranking_metrics" in first_cfg
    assert "strategy_metrics" in first_cfg
