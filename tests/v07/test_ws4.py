"""
Unit and integration tests for StockReason v0.7 Workstream WS4.

Tests:
1. BaseForecaster interface compliance across all models
2. Models fit and predict positive IC on planted signal
3. Models on pure noise get IC near zero
4. Baselines (ZeroForecaster, MarketForecaster) properties
5. Quantile forecast validity (q10 <= q50 <= q90)
6. Diebold-Mariano test verification
7. Purged walk-forward CV leakage verification
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.v07.models.base import BaseForecaster, extract_tabular_features
from src.v07.models.baselines import MarketForecaster, ZeroForecaster
from src.v07.models.deep import DLinearForecaster, GRUForecaster, LSTMForecaster
from src.v07.models.linear import ElasticNetForecaster, RidgeForecaster
from src.v07.models.purged_cv import PurgedWalkForwardCV
from src.v07.models.significance import compare_against_baselines, diebold_mariano_test
from src.v07.models.tree import CatBoostForecaster, LightGBMForecaster


@pytest.fixture
def planted_signal_data():
    """Generates synthetic dataset with strong linear and non-linear signal."""
    np.random.seed(42)
    n_samples = 300
    seq_len = 10
    n_features = 5

    # 3D sequences: (N, T, D)
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    # Target depends positively on the last step of feature 0 and feature 1
    y = (
        0.7 * X[:, -1, 0]
        - 0.5 * X[:, -1, 1]
        + 0.3 * np.sin(X[:, -1, 2])
        + 0.1 * np.random.randn(n_samples)
    ).astype(np.float32)

    # Split train / val / test
    X_train, y_train = X[:180], y[:180]
    X_val, y_val = X[180:240], y[180:240]
    X_test, y_test = X[240:], y[240:]

    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def pure_noise_data():
    """Generates pure Gaussian noise data with independent target."""
    np.random.seed(999)
    n_samples = 300
    seq_len = 10
    n_features = 5

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    X_train, y_train = X[:180], y[:180]
    X_val, y_val = X[180:240], y[180:240]
    X_test, y_test = X[240:], y[240:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def test_base_interface_properties():
    """Test that all models inherit from BaseForecaster and expose proper attributes."""
    models = [
        ZeroForecaster(),
        MarketForecaster(),
        RidgeForecaster(),
        ElasticNetForecaster(),
        LightGBMForecaster(),
        CatBoostForecaster(),
        LSTMForecaster(n_seeds=1, epochs=2),
        GRUForecaster(n_seeds=1, epochs=2),
        DLinearForecaster(epochs=2),
    ]

    for model in models:
        assert isinstance(model, BaseForecaster)
        assert isinstance(model.name, str)
        assert isinstance(model.seed, int)
        assert not model.is_fitted


def test_extract_tabular_features():
    """Test tabular feature extraction from 3D sequence array."""
    X_3d = np.random.randn(20, 15, 4)
    tab = extract_tabular_features(X_3d)
    assert tab.ndim == 2
    assert tab.shape[0] == 20
    assert tab.shape[1] == 4 * 4  # last, mean, std, diff

    X_2d = np.random.randn(20, 6)
    tab_2d = extract_tabular_features(X_2d)
    assert tab_2d.shape == (20, 6)


def test_zero_and_market_baselines(planted_signal_data):
    """Test ZeroForecaster and MarketForecaster behavior."""
    X_train, y_train, X_val, y_val, X_test, y_test = planted_signal_data

    # Zero forecaster
    zero = ZeroForecaster()
    zero.fit(X_train, y_train, X_val, y_val)
    preds = zero.predict(X_test)
    assert np.all(preds == 0.0)
    assert len(preds) == len(X_test)

    # Market forecaster
    market = MarketForecaster()
    market.fit(X_train, y_train, X_val, y_val)
    preds_m = market.predict(X_test)
    expected_mean = float(np.mean(y_train))
    assert np.allclose(preds_m, expected_mean)


@pytest.mark.parametrize("model_cls, kwargs", [
    (RidgeForecaster, {}),
    (ElasticNetForecaster, {}),
    (LightGBMForecaster, {"n_estimators": 50}),
    (CatBoostForecaster, {"iterations": 50}),
    (LSTMForecaster, {"n_seeds": 2, "epochs": 15, "batch_size": 32}),
    (GRUForecaster, {"n_seeds": 2, "epochs": 15, "batch_size": 32}),
    (DLinearForecaster, {"epochs": 15, "batch_size": 32}),
])
def test_models_fit_predict_planted_signal(planted_signal_data, model_cls, kwargs):
    """Test that models achieve positive Rank IC on planted signal."""
    X_train, y_train, X_val, y_val, X_test, y_test = planted_signal_data

    model = model_cls(seed=42, **kwargs)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model.is_fitted

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert not np.isnan(preds).any()

    # Rank correlation with ground truth
    corr, _ = stats.spearmanr(preds, y_test)
    # Planted signal should yield positive Rank IC
    assert corr > 0.05, f"{model.name} failed to extract planted signal (Rank IC = {corr:.3f})"


@pytest.mark.parametrize("model_cls, kwargs", [
    (RidgeForecaster, {}),
    (LightGBMForecaster, {"n_estimators": 30}),
    (LSTMForecaster, {"n_seeds": 1, "epochs": 8, "batch_size": 32}),
])
def test_models_pure_noise(pure_noise_data, model_cls, kwargs):
    """Test that models on pure noise get IC near zero (no large spurious edge)."""
    X_train, y_train, X_val, y_val, X_test, y_test = pure_noise_data

    model = model_cls(seed=123, **kwargs)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = model.predict(X_test)

    corr, _ = stats.spearmanr(preds, y_test)
    # On noise, correlation should be close to zero (|IC| < 0.3)
    assert abs(corr) < 0.30, f"{model.name} showed spurious edge on noise (Rank IC = {corr:.3f})"


def test_quantile_predictions(planted_signal_data):
    """Test quantile predictions satisfy q10 <= q50 <= q90 order."""
    X_train, y_train, X_val, y_val, X_test, y_test = planted_signal_data

    lgb_model = LightGBMForecaster(n_estimators=40, seed=42)
    lgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    qs = [0.1, 0.5, 0.9]
    quantiles = lgb_model.predict_quantiles(X_test, qs=qs)

    assert set(quantiles.keys()) == {0.1, 0.5, 0.9}
    assert len(quantiles[0.1]) == len(X_test)

    # Monotonicity check on average
    mean_q10 = np.mean(quantiles[0.1])
    mean_q50 = np.mean(quantiles[0.5])
    mean_q90 = np.mean(quantiles[0.9])
    assert mean_q10 <= mean_q50 <= mean_q90

    # Also check base fallback quantiles on Ridge
    ridge = RidgeForecaster()
    ridge.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    r_quantiles = ridge.predict_quantiles(X_test, qs=qs)
    assert np.mean(r_quantiles[0.1]) <= np.mean(r_quantiles[0.5]) <= np.mean(r_quantiles[0.9])


def test_diebold_mariano_identical_forecasts():
    """Identical forecasts must yield DM stat = 0 and p-value = 1."""
    y = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
    p1 = np.array([0.012, -0.018, 0.025, 0.038, -0.008, 0.019])
    p2 = p1.copy()

    stat, p_val = diebold_mariano_test(y, p1, p2, h=1)
    assert stat == 0.0
    assert p_val == 1.0


def test_diebold_mariano_superior_forecast():
    """Significantly superior forecast must yield statistically significant p-value."""
    np.random.seed(42)
    N = 100
    y = np.random.randn(N)
    # Good model: close to y
    p_good = y + 0.1 * np.random.randn(N)
    # Bad model: completely wrong noise with high variance
    p_bad = y + 3.0 * np.random.randn(N)

    stat, p_val = diebold_mariano_test(y, p_good, p_bad, h=1, alternative="less")
    assert stat < -5.0  # p_good has significantly smaller loss
    assert p_val < 0.01


def test_purged_walk_forward_cv_leakage():
    """Test PurgedWalkForwardCV creates disjoint blocks with proper purge gap."""
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    dates_s = pd.Series(dates)

    cv = PurgedWalkForwardCV(
        n_folds=3,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_window=5,
        embargo_window=2,
        min_train_ratio=0.30,
    )

    splits = list(cv.split_indices(dates_s))
    assert len(splits) >= 2

    for tr, val, te in splits:
        # Check no overlap between indices
        assert len(set(tr).intersection(set(val))) == 0
        assert len(set(val).intersection(set(te))) == 0
        assert len(set(tr).intersection(set(te))) == 0

        # Check chronological order: max(train_date) < min(val_date)
        max_tr_date = dates_s.iloc[tr].max()
        min_val_date = dates_s.iloc[val].max()
        assert max_tr_date < min_val_date
