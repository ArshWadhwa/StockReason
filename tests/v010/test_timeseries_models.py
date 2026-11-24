"""
Unit tests for StockReason v0.10 Time-Series Architectures (DLinear, TCN, N-HiTS).
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.v010.timeseries.models import (
    DLinearModel, TCNModel, NHiTSModel, evaluate_timeseries_architectures
)


def test_timeseries_models_forward_and_evaluation():
    batch_size = 16
    seq_len = 64
    n_features = 21

    x = torch.randn(batch_size, seq_len, n_features)

    # 1. DLinear
    dlinear = DLinearModel(seq_len=seq_len, n_features=n_features)
    out_dlinear = dlinear(x)
    assert out_dlinear.shape == (batch_size,)

    # 2. TCN
    tcn = TCNModel(n_features=n_features, hidden_dim=32)
    out_tcn = tcn(x)
    assert out_tcn.shape == (batch_size,)

    # 3. N-HiTS
    nhits = NHiTSModel(seq_len=seq_len, n_features=n_features)
    out_nhits = nhits(x)
    assert out_nhits.shape == (batch_size,)

    # Test training & evaluation harness
    X_train_seq = np.random.randn(200, seq_len, n_features).astype(np.float32)
    y_train = np.random.randn(200).astype(np.float32) * 0.05
    X_val_seq = np.random.randn(50, seq_len, n_features).astype(np.float32)
    y_val = np.random.randn(50).astype(np.float32) * 0.05
    val_close = np.full(50, 100.0)

    res_df = evaluate_timeseries_architectures(
        X_train_seq, y_train, X_val_seq, y_val, val_close,
        batch_size=64, epochs=2
    )

    assert len(res_df) == 3
    assert set(res_df["architecture"].values) == {"DLinear", "TCN", "N-HiTS"}
    assert all(res_df["return_mae"] > 0)
