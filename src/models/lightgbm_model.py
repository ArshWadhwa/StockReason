"""LightGBM Native Model Implementation & Wrapper."""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LightGBMForecaster:
    """
    Production LightGBM Forecaster supporting native text serialization (.txt),
    robust L1 regression objective, and early stopping.
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        num_leaves: int = 31,
        objective: str = "regression_l1",
    ):
        if lgb is None:
            raise ImportError("lightgbm is not installed.")
        self.feature_names = feature_names or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.objective = objective
        self.booster_: Optional[lgb.Booster] = None
        self.model_: Optional[lgb.LGBMRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ) -> "LightGBMForecaster":
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            objective=self.objective,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)] if eval_set else None

        self.model_.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        self.booster_ = self.model_.booster_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.booster_ is not None:
            return self.booster_.predict(X)
        elif self.model_ is not None:
            return self.model_.predict(X)
        raise RuntimeError("Model is not fitted or loaded.")

    def save_model(self, path: Union[str, Path]):
        """Saves native LightGBM text model (.txt)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.booster_ is not None:
            self.booster_.save_model(str(path))
        elif self.model_ is not None:
            self.model_.booster_.save_model(str(path))
        else:
            raise RuntimeError("No model to save.")

    def load_model(self, path: Union[str, Path]) -> "LightGBMForecaster":
        """Loads native LightGBM text model (.txt)."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.booster_ = lgb.Booster(model_file=str(path))
        self.feature_names = self.booster_.feature_name()
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        if self.booster_ is None:
            return {}
        imp = self.booster_.feature_importance(importance_type="gain")
        names = self.booster_.feature_name()
        return {name: float(val) for name, val in zip(names, imp)}
