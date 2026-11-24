"""
Tree-based gradient boosted forecasters for StockReason v0.7.

Implements:
- LightGBMForecaster: LightGBM regressor with early stopping on validation block,
  and native quantile regression variants (q10, q50, q90).
- CatBoostForecaster: CatBoost regressor with early stopping on validation block.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    from src.v07.models.base import BaseForecaster, extract_tabular_features
except ImportError:
    from v07.models.base import BaseForecaster, extract_tabular_features


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM gradient boosting forecaster.
    Supports early stopping on validation set and native quantile regression.
    """

    def __init__(
        self,
        name: str = "lightgbm",
        seed: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.LGBMRegressor] = None
        self.quantile_models: Dict[float, lgb.LGBMRegressor] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMForecaster":
        X_tr = extract_tabular_features(X_train)
        y_tr = np.asarray(y_train).flatten()

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            y_v = np.asarray(y_val).flatten()
            eval_set = [(X_v, y_v)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        # 1. Main regression model
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
        )

        if eval_set is not None:
            self.model.fit(
                X_tr,
                y_tr,
                eval_set=eval_set,
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_tr, y_tr)

        train_pred = self.model.predict(X_tr)
        self.train_residuals = y_tr - train_pred

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            y_v = np.asarray(y_val).flatten()
            val_pred = self.model.predict(X_v)
            self.val_residuals = y_v - val_pred

        # 2. Fit quantile models for q10, q50, q90
        for q in [0.1, 0.5, 0.9]:
            q_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=min(self.n_estimators, 150),
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.seed + int(q * 100),
                n_jobs=-1,
                verbosity=-1,
            )
            if eval_set is not None:
                q_model.fit(X_tr, y_tr, eval_set=eval_set, callbacks=callbacks)
            else:
                q_model.fit(X_tr, y_tr)
            self.quantile_models[q] = q_model

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_tab = extract_tabular_features(X)
        return self.model.predict(X_tab)

    def predict_quantiles(
        self, X: np.ndarray, qs: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_tab = extract_tabular_features(X)
        out: Dict[float, np.ndarray] = {}
        for q in qs:
            if q in self.quantile_models:
                out[q] = self.quantile_models[q].predict(X_tab)
            else:
                # Fallback to residual offset
                residuals = self.val_residuals if self.val_residuals is not None else self.train_residuals
                point_pred = self.predict(X)
                if residuals is not None and len(residuals) > 0:
                    offset = float(np.percentile(residuals, q * 100))
                    out[q] = point_pred + offset
                else:
                    out[q] = point_pred.copy()
        return out


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost gradient boosting forecaster.
    Supports early stopping on validation set.
    """

    def __init__(
        self,
        name: str = "catboost",
        seed: int = 42,
        iterations: int = 300,
        learning_rate: float = 0.03,
        depth: int = 6,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[CatBoostRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CatBoostForecaster":
        X_tr = extract_tabular_features(X_train)
        y_tr = np.asarray(y_train).flatten()

        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            y_v = np.asarray(y_val).flatten()
            eval_set = (X_v, y_v)

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.seed,
            early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None,
            verbose=0,
        )

        if eval_set is not None:
            self.model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_tr, y_tr, verbose=False)

        train_pred = self.model.predict(X_tr)
        self.train_residuals = y_tr - train_pred

        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = extract_tabular_features(X_val)
            y_v = np.asarray(y_val).flatten()
            val_pred = self.model.predict(X_v)
            self.val_residuals = y_v - val_pred

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_tab = extract_tabular_features(X)
        return self.model.predict(X_tab)
