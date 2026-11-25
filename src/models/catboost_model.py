"""CatBoost Native Model Implementation & Wrapper."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


class CatBoostForecaster:
    """
    Production CatBoost Forecaster supporting native binary serialization (.cbm)
    and robust MAE loss function.
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        iterations: int = 300,
        learning_rate: float = 0.03,
        depth: int = 6,
        loss_function: str = "MAE",
    ):
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        self.feature_names = feature_names or []
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.model_: Optional[CatBoostRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ) -> "CatBoostForecaster":
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            verbose=0,
            random_seed=42,
            thread_count=4,
        )
        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None
        self.model_.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted or loaded.")
        return self.model_.predict(X)

    def save_model(self, path: Union[str, Path]):
        """Saves native CatBoost binary model (.cbm)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model_ is not None:
            self.model_.save_model(str(path))
        else:
            raise RuntimeError("No model to save.")

    def load_model(self, path: Union[str, Path]) -> "CatBoostForecaster":
        """Loads native CatBoost binary model (.cbm)."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model_ = CatBoostRegressor()
        self.model_.load_model(str(path))
        if hasattr(self.model_, "feature_names_") and self.model_.feature_names_:
            self.feature_names = list(self.model_.feature_names_)
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        if self.model_ is None:
            return {}
        imp = self.model_.get_feature_importance()
        names = self.feature_names or [f"f_{i}" for i in range(len(imp))]
        return {name: float(val) for name, val in zip(names, imp)}
