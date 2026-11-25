"""Baseline Plus Residual Learning Architecture."""

from typing import Tuple, Any, Optional
import numpy as np
import pandas as pd


class BaselinePlusResidualLearningEngine:
    """
    Implements decomposition:
    y = Baseline(x) + ResidualModel(x)
    hat{y} = Baseline(x) + alpha * ResidualModel(x)
    where alpha in [0, 1] is shrinkage tuned on validation fold.
    """
    def __init__(self, baseline_model: Any, residual_model: Any, alpha: float = 1.0):
        self.baseline_model = baseline_model
        self.residual_model = residual_model
        self.alpha = alpha

    def fit_residual(
        self,
        X_train: np.ndarray,
        y_train_return: np.ndarray,
        base_train_return: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_return: Optional[np.ndarray] = None,
        base_val_return: Optional[np.ndarray] = None,
    ):
        residual_train = (y_train_return - base_train_return).astype(np.float32)
        residual_val = (y_val_return - base_val_return).astype(np.float32) if y_val_return is not None else None

        self.residual_model.fit(
            X_train, residual_train,
            X_val=X_val, y_val=residual_val,
        )

    def tune_alpha(
        self,
        X_val: np.ndarray,
        y_val_price: np.ndarray,
        base_val_return: np.ndarray,
        val_close: np.ndarray,
        grid: Optional[list] = None,
    ) -> float:
        if grid is None:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        pred_res = self.residual_model.predict(X_val)
        best_a = self.alpha
        best_mape = float("inf")

        for a in grid:
            ret_cand = base_val_return + a * pred_res
            p_cand = val_close * (1.0 + ret_cand)
            pe = np.abs(y_val_price - p_cand) / np.maximum(y_val_price, 0.01)
            mape = float(np.mean(pe) * 100.0)
            if mape < best_mape:
                best_mape = mape
                best_a = a

        self.alpha = best_a
        return self.alpha

    def predict(
        self,
        X: np.ndarray,
        base_return: np.ndarray,
        close: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        res = self.residual_model.predict(X)
        pred_return = base_return + self.alpha * res
        pred_price = close * (1.0 + pred_return)
        return pred_return, pred_price
