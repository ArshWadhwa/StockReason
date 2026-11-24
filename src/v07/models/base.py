"""
Base forecaster interface for StockReason v0.7.

Defines the abstract BaseForecaster contract for all tabular and sequence models:
- fit(X_train, y_train, X_val=None, y_val=None)
- predict(X)
- predict_quantiles(X, qs=[0.1, 0.5, 0.9])
- name and seed properties
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional, Union
import numpy as np


def extract_tabular_features(X: np.ndarray) -> np.ndarray:
    """
    Transforms 3D sequence array (N, T, D) into a 2D tabular feature vector (N, D_tab)
    using the last time step plus summary statistics across time.
    If X is already 2D (N, D), returns X unchanged.

    Summary statistics included:
    - Last-step feature vector: X[:, -1, :]
    - Mean across time window: X.mean(axis=1)
    - Std across time window: X.std(axis=1)
    - Total change: X[:, -1, :] - X[:, 0, :]
    """
    if X.ndim == 2:
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    elif X.ndim == 3:
        n_samples, seq_len, n_features = X.shape
        last_step = X[:, -1, :]
        mean_step = np.mean(X, axis=1)
        std_step = np.std(X, axis=1)
        diff_step = X[:, -1, :] - X[:, 0, :]
        tab_feats = np.hstack([last_step, mean_step, std_step, diff_step])
        return np.nan_to_num(tab_feats, nan=0.0, posinf=0.0, neginf=0.0)
    elif X.ndim == 1:
        return np.nan_to_num(X.reshape(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"Unsupported input dimension for X: {X.ndim}")


class BaseForecaster(abc.ABC):
    """
    Common interface for all StockReason v0.7 forecasting models.
    """

    def __init__(self, name: str, seed: int = 42, **kwargs) -> None:
        self._name = name
        self._seed = seed
        self.is_fitted = False
        self.val_residuals: Optional[np.ndarray] = None
        self.train_residuals: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        """Name of the forecaster."""
        return self._name

    @property
    def seed(self) -> int:
        """Random seed for reproducibility."""
        return self._seed

    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseForecaster":
        """
        Fit model on training set with optional validation set for early stopping
        or hyperparameter tuning.

        Parameters
        ----------
        X_train : np.ndarray
            Features array (2D tabular or 3D sequence).
        y_train : np.ndarray
            1D target returns array.
        X_val : Optional[np.ndarray]
            Validation features.
        y_val : Optional[np.ndarray]
            Validation target returns.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions for input features X.

        Returns
        -------
        np.ndarray
            1D array of predicted returns of length N.
        """
        pass

    def predict_quantiles(
        self, X: np.ndarray, qs: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        """
        Generate quantile forecasts for input features X.
        Default implementation estimates quantiles using empirical residuals
        from training / validation data. Models with native quantile regression
        override this method.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        qs : List[float]
            Quantile levels in (0, 1).

        Returns
        -------
        Dict[float, np.ndarray]
            Dictionary mapping quantile q -> 1D array of predictions at level q.
        """
        point_pred = self.predict(X)
        residuals = self.val_residuals if self.val_residuals is not None else self.train_residuals

        out: Dict[float, np.ndarray] = {}
        if residuals is not None and len(residuals) > 0:
            for q in qs:
                offset = float(np.percentile(residuals, q * 100))
                out[q] = point_pred + offset
        else:
            for q in qs:
                out[q] = point_pred.copy()
        return out

    def predict_prob_up(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the probability of positive return P(y > 0 | X).
        Uses quantile / residual distribution or normal assumption.
        """
        point_pred = self.predict(X)
        residuals = self.val_residuals if self.val_residuals is not None else self.train_residuals
        if residuals is not None and len(residuals) > 1:
            std = float(np.std(residuals))
            if std > 1e-8:
                # Normal approximation: P(Y > 0) = 1 - Phi(-pred / std) = Phi(pred / std)
                from scipy.stats import norm
                return norm.cdf(point_pred / std)

        # Fallback binary sign heuristic
        return (point_pred > 0).astype(float)
