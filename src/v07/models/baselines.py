"""
Baseline forecasters for StockReason v0.7.

Implements:
- ZeroForecaster: predicts 0 excess return.
- MarketForecaster: predicts trailing-mean market return.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

try:
    from src.v07.models.base import BaseForecaster, extract_tabular_features
except ImportError:
    from v07.models.base import BaseForecaster, extract_tabular_features


class ZeroForecaster(BaseForecaster):
    """
    Zero excess return forecaster.
    Predicts constant 0.0 for all samples. Serves as the primary null baseline.
    """

    def __init__(self, name: str = "zero", seed: int = 42, **kwargs) -> None:
        super().__init__(name=name, seed=seed)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ZeroForecaster":
        y_train_flat = np.asarray(y_train).flatten()
        self.train_residuals = y_train_flat - 0.0
        if y_val is not None:
            self.val_residuals = np.asarray(y_val).flatten() - 0.0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.zeros(n_samples, dtype=float)

    def predict_quantiles(
        self, X: np.ndarray, qs: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        n_samples = len(X)
        residuals = self.val_residuals if self.val_residuals is not None else self.train_residuals
        out: Dict[float, np.ndarray] = {}
        if residuals is not None and len(residuals) > 0:
            for q in qs:
                val = float(np.percentile(residuals, q * 100))
                out[q] = np.full(n_samples, val, dtype=float)
        else:
            for q in qs:
                out[q] = np.zeros(n_samples, dtype=float)
        return out


class MarketForecaster(BaseForecaster):
    """
    Market return forecaster.
    Predicts the trailing-mean return from training block.
    If market return feature is provided, uses its mean; otherwise uses mean of y_train.
    """

    def __init__(self, name: str = "market", seed: int = 42, **kwargs) -> None:
        super().__init__(name=name, seed=seed)
        self.trailing_mean: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MarketForecaster":
        y_arr = np.asarray(y_train).flatten()
        self.trailing_mean = float(np.nanmean(y_arr)) if len(y_arr) > 0 else 0.0

        train_pred = np.full_like(y_arr, self.trailing_mean)
        self.train_residuals = y_arr - train_pred

        if y_val is not None:
            y_val_arr = np.asarray(y_val).flatten()
            val_pred = np.full_like(y_val_arr, self.trailing_mean)
            self.val_residuals = y_val_arr - val_pred

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.trailing_mean, dtype=float)

    def predict_quantiles(
        self, X: np.ndarray, qs: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        n_samples = len(X)
        residuals = self.val_residuals if self.val_residuals is not None else self.train_residuals
        out: Dict[float, np.ndarray] = {}
        if residuals is not None and len(residuals) > 0:
            for q in qs:
                offset = float(np.percentile(residuals, q * 100))
                out[q] = np.full(n_samples, self.trailing_mean + offset, dtype=float)
        else:
            for q in qs:
                out[q] = np.full(n_samples, self.trailing_mean, dtype=float)
        return out
