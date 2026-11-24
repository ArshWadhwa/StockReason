"""
Unit tests for StockReason v0.8 Feature Engineering and Multi-Horizon Pipeline Integration.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import pytest
import pandas as pd
import numpy as np

from src.v08.features import (
    compute_technical_features,
    compute_multi_horizon_targets,
    extract_feature_column_names,
    SUPPORTED_HORIZONS,
)
from src.v08.pipeline_integration import (
    PurgedWalkForwardCV,
    train_and_evaluate_model,
)


def create_mock_canonical_panel(n_tickers: int = 10, n_days: int = 300) -> pd.DataFrame:
    """Helper creating multi-ticker synthetic panel."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    records = []
    for i in range(n_tickers):
        ticker = f"STOCK_{i}.NS"
        base_p = 500.0 + i * 50.0
        trend = np.linspace(0, 50, n_days)
        noise = np.sin(np.linspace(0, 10, n_days) + i) * 10.0
        close_p = base_p + trend + noise
        for d_idx, dt in enumerate(dates):
            cp = close_p[d_idx]
            records.append({
                "ticker": ticker,
                "date": dt,
                "open": cp - 2.0,
                "high": cp + 5.0,
                "low": cp - 4.0,
                "close": cp,
                "adj_close": cp * 0.95,
                "volume": 100000 + int(i * 1000 + noise[d_idx] * 500),
            })
    return pd.DataFrame(records)


def test_technical_features_generation():
    panel = create_mock_canonical_panel(n_tickers=3, n_days=100)
    feat_df = compute_technical_features(panel)

    expected_cols = ["return_1d", "return_21d", "vol_20d", "rsi_14", "macd", "dist_sma_20"]
    for c in expected_cols:
        assert c in feat_df.columns, f"Expected feature {c} missing in output"

    feature_names = extract_feature_column_names(feat_df)
    assert len(feature_names) >= 15


def test_anti_leakage_truncation_invariance():
    """
    CRITICAL ANTI-LEAKAGE INVARIANT:
    Features computed on the full dataset up to date t must match EXACTLY
    with features computed on truncated history ending at date t.
    """
    panel = create_mock_canonical_panel(n_tickers=1, n_days=120)
    full_feat = compute_technical_features(panel)

    # Pick test date at day 80
    test_date = panel["date"].iloc[80]

    # Truncate raw panel to day 80 and compute features
    truncated_raw = panel[panel["date"] <= test_date].copy()
    trunc_feat = compute_technical_features(truncated_raw)

    feature_cols = extract_feature_column_names(full_feat)

    row_full = full_feat[full_feat["date"] == test_date][feature_cols].iloc[0]
    row_trunc = trunc_feat[trunc_feat["date"] == test_date][feature_cols].iloc[0]

    for col in feature_cols:
        val_full = row_full[col]
        val_trunc = row_trunc[col]
        if not (np.isnan(val_full) and np.isnan(val_trunc)):
            diff = abs(val_full - val_trunc)
            assert diff < 1e-5, f"Lookahead bias detected in {col}: full={val_full}, trunc={val_trunc}"


def test_multi_horizon_targets():
    panel = create_mock_canonical_panel(n_tickers=5, n_days=200)
    target_df = compute_multi_horizon_targets(panel, horizons=[1, 5, 21, 63, 126])

    for h in [1, 5, 21, 63, 126]:
        assert f"target_return_{h}d" in target_df.columns
        assert f"target_price_{h}d" in target_df.columns
        assert f"target_direction_{h}d" in target_df.columns
        assert f"target_rank_{h}d" in target_df.columns

    # Arithmetic verification on random row
    row_idx = 10
    h = 21
    curr_adj = panel["adj_close"].iloc[row_idx]
    fut_adj = panel["adj_close"].iloc[row_idx + h]
    expected_ret = (fut_adj - curr_adj) / curr_adj

    actual_ret = target_df[target_df["ticker"] == panel["ticker"].iloc[row_idx]].iloc[row_idx][f"target_return_{h}d"]
    assert abs(actual_ret - expected_ret) < 1e-4


def test_purged_walk_forward_cv_horizons():
    dates = pd.date_range("2020-01-01", periods=500, freq="B")

    # Horizon 21d -> purge 26 trading days
    cv_21 = PurgedWalkForwardCV(dates, n_folds=5, horizon=21, embargo_window=5)
    splits_21 = cv_21.split()
    assert len(splits_21) > 0

    for split in splits_21:
        # Strict non-overlap
        train_s = set(split.train_dates)
        val_s = set(split.val_dates)
        test_s = set(split.test_dates)
        assert len(train_s.intersection(val_s)) == 0
        assert len(val_s.intersection(test_s)) == 0
        assert len(train_s.intersection(test_s)) == 0

        # Purge window has exactly 26 trading days between train and val
        max_train = max(split.train_dates)
        min_val = min(split.val_dates)
        purged_trading_days = [d for d in dates if max_train < d < min_val]
        assert len(purged_trading_days) == 26

    # Horizon 126d -> purge 131 trading days (tested on 1000 days ~ 4 years)
    dates_long = pd.date_range("2015-01-01", periods=1000, freq="B")
    cv_126 = PurgedWalkForwardCV(dates_long, n_folds=3, horizon=126, embargo_window=5)
    splits_126 = cv_126.split()
    assert len(splits_126) > 0
    for split in splits_126:
        max_train = max(split.train_dates)
        min_val = min(split.val_dates)
        purged_trading_days = [d for d in dates_long if max_train < d < min_val]
        assert len(purged_trading_days) == 131


def test_train_and_evaluate_models():
    # Synthetic dataset for model integration test
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

    dates = np.array([f"2024-01-{i%20 + 1:02d}" for i in range(n_samples)])
    tickers = np.array([f"STOCK_{i%10}.NS" for i in range(n_samples)])

    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:150], y[120:150]
    X_test, y_test = X[150:], y[150:]
    te_dates = dates[150:]
    te_tickers = tickers[150:]

    for m in ["ridge", "lightgbm", "catboost", "gru"]:
        res = train_and_evaluate_model(
            model_name=m,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_dates=te_dates,
            test_tickers=te_tickers,
            horizon=21,
        )
        assert res["model"] == m
        assert "mae" in res
        assert "directional_accuracy_pct" in res
        assert "mean_rank_ic" in res
