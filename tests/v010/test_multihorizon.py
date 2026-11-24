"""
Unit tests for StockReason v0.10 Multi-Horizon & Multi-Task Neural Learning.
"""

import pytest
import numpy as np
import pandas as pd

from src.v010.multihorizon.multitask_net import train_multitask_neural_network, HORIZONS
from src.v010.multihorizon.hierarchical import compare_independent_vs_multitask, verify_hierarchical_horizon_consistency


def test_multitask_network_training_and_comparison():
    np.random.seed(42)
    n_samples = 500
    n_features = 21

    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(100, n_features)
    val_close = 100.0 + np.random.rand(100) * 50.0

    y_train = {}
    y_val = {}
    for h in HORIZONS:
        scale = np.sqrt(h / 252.0)
        y_train[h] = np.random.randn(n_samples) * scale
        y_val[h] = np.random.randn(100) * scale

    # Mask a few values
    y_train[21][10:15] = np.nan

    res = train_multitask_neural_network(
        X_train, y_train, X_val, y_val, val_close,
        batch_size=128, epochs=2, sample_limit=500
    )

    assert "metrics_table" in res
    assert "predictions" in res
    assert len(res["metrics_table"]) == len(HORIZONS)
    assert all(res["metrics_table"]["return_mae"] > 0)

    # Compare independent vs multitask
    comp_df = compare_independent_vs_multitask(X_train, y_train, X_val, y_val, val_close, res)
    assert len(comp_df) == len(HORIZONS) * 2
    assert "Approach A: Independent Ridge" in comp_df["approach"].values
    assert "Approach B: Multi-Task Neural Net" in comp_df["approach"].values

    # Hierarchical consistency
    hier = verify_hierarchical_horizon_consistency(res["predictions"])
    assert "prediction_std_by_horizon" in hier
    assert len(hier["prediction_std_by_horizon"]) == len(HORIZONS)
