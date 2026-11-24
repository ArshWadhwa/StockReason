"""
Linear baseline forecasters with time-aware validation tuning for StockReason v0.7.

Implements:
- RidgeForecaster: Ridge regression with alpha tuning on validation block.
- ElasticNetForecaster: ElasticNet regression with alpha and l1_ratio tuning on validation block.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

try:
    from src.v07.models.base import BaseForecaster, extract_tabular_features
except ImportError:
    from v07.models.base import BaseForecaster, extract_tabular_features


class RidgeForecaster(BaseForecaster):
    """
    Ridge regression forecaster.
    Fits StandardScaler on X_train only.
    Tuning: evaluates a grid of alpha penalties on the chronological validation block
    to pick the optimal regularization strength without lookahead leakage.
    """

    def __init__(
        self,
        name: str = "ridge",
        seed: int = 42,
        alphas: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.alphas = alphas if alphas is not None else [
            0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0
        ]
        self.best_alpha: float = 1.0
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Ridge] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RidgeForecaster":
        X_tr = extract_tabular_features(X_train)
        y_tr = np.asarray(y_train).flatten()

        # Fit scaler ONLY on train to avoid leakage
        self.scaler = StandardScaler()
        X_tr_scaled = self.scaler.fit_transform(X_tr)

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            X_v_scaled = self.scaler.transform(X_v)
            y_v = np.asarray(y_val).flatten()

            best_mse = float("inf")
            best_a = self.alphas[0]
            for alpha in self.alphas:
                ridge_cand = Ridge(alpha=alpha, random_state=self.seed)
                ridge_cand.fit(X_tr_scaled, y_tr)
                val_pred = ridge_cand.predict(X_v_scaled)
                mse = float(np.mean((val_pred - y_v) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_a = alpha
            self.best_alpha = best_a
        else:
            self.best_alpha = 1.0

        # Fit final model with selected best_alpha
        self.model = Ridge(alpha=self.best_alpha, random_state=self.seed)
        self.model.fit(X_tr_scaled, y_tr)

        train_pred = self.model.predict(X_tr_scaled)
        self.train_residuals = y_tr - train_pred

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            X_v_scaled = self.scaler.transform(X_v)
            y_v = np.asarray(y_val).flatten()
            val_pred = self.model.predict(X_v_scaled)
            self.val_residuals = y_v - val_pred

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_tab = extract_tabular_features(X)
        X_scaled = self.scaler.transform(X_tab)
        return self.model.predict(X_scaled)


class ElasticNetForecaster(BaseForecaster):
    """
    ElasticNet regression forecaster (L1 + L2 regularization).
    Fits StandardScaler on X_train only.
    Tuning: grid search over alpha and l1_ratio on validation block.
    """

    def __init__(
        self,
        name: str = "elastic_net",
        seed: int = 42,
        alphas: Optional[List[float]] = None,
        l1_ratios: Optional[List[float]] = None,
        max_iter: int = 2000,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.alphas = alphas if alphas is not None else [0.001, 0.01, 0.1, 1.0, 10.0]
        self.l1_ratios = l1_ratios if l1_ratios is not None else [0.1, 0.5, 0.7, 0.9]
        self.max_iter = max_iter
        self.best_alpha: float = 0.01
        self.best_l1_ratio: float = 0.5
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[ElasticNet] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ElasticNetForecaster":
        X_tr = extract_tabular_features(X_train)
        y_tr = np.asarray(y_train).flatten()

        self.scaler = StandardScaler()
        X_tr_scaled = self.scaler.fit_transform(X_tr)

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            X_v_scaled = self.scaler.transform(X_v)
            y_v = np.asarray(y_val).flatten()

            best_mse = float("inf")
            best_a = self.alphas[0]
            best_l1 = self.l1_ratios[0]
            for alpha in self.alphas:
                for l1_ratio in self.l1_ratios:
                    en_cand = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=self.max_iter,
                        random_state=self.seed,
                    )
                    en_cand.fit(X_tr_scaled, y_tr)
                    val_pred = en_cand.predict(X_v_scaled)
                    mse = float(np.mean((val_pred - y_v) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_a = alpha
                        best_l1 = l1_ratio
            self.best_alpha = best_a
            self.best_l1_ratio = best_l1
        else:
            self.best_alpha = 0.01
            self.best_l1_ratio = 0.5

        self.model = ElasticNet(
            alpha=self.best_alpha,
            l1_ratio=self.best_l1_ratio,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(X_tr_scaled, y_tr)

        train_pred = self.model.predict(X_tr_scaled)
        self.train_residuals = y_tr - train_pred

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            X_v_scaled = self.scaler.transform(X_v)
            y_v = np.asarray(y_val).flatten()
            val_pred = self.model.predict(X_v_scaled)
            self.val_residuals = y_v - val_pred

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_tab = extract_tabular_features(X)
        X_scaled = self.scaler.transform(X_tab)
        return self.model.predict(X_scaled)
