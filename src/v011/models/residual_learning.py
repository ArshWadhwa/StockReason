"""
StockReason v0.11 — Baseline Plus Residual Learning Engine.
===========================================================
Remediates Phase 4 (Section 24):
1. Principle:
   Rather than asking an unconstrained ML model to predict the full future return
   from scratch, anchor to the strongest validated statistical baseline (Historical Mean Return),
   and learn a small, regularized stock-specific residual correction:
   Final predicted return = Baseline forecast + alpha * ML residual correction
2. Target:
   residual_target = actual_future_return - baseline_forecast
3. Candidate Residual Learners:
   - Ridge Residual Regression (L2 regularized)
   - ElasticNet Residual Regression (L1 + L2)
   - Robust Huber Residual Regression (outlier-insensitive)
   - LightGBM Residual Regressor (Gradient Boosted Trees on residuals)
   - CatBoost Residual Regressor (Categorical handling on residuals)
4. Shrinkage Optimization:
   Learns alpha in [0.0, 1.0] strictly on development fold. If alpha=0, falls back cleanly to baseline.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.v011.evaluation.baselines import compute_price_metrics


class BaselinePlusResidualLearningEngine:
    def __init__(
        self,
        baseline_type: str = "historical_mean",
        residual_model_type: str = "lgbm",
        horizon: int = 21,
        seed: int = 42,
    ):
        self.baseline_type = baseline_type
        self.residual_model_type = residual_model_type
        self.horizon = horizon
        self.seed = seed
        self.baseline_val: float = 0.0
        self.residual_model = None
        self.alpha: float = 1.0

    def fit(
        self,
        train_df: pd.DataFrame,
        features: List[str],
        dev_df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
    ):
        """
        Fits baseline, fits residual model on (y_actual - baseline),
        and tunes shrinkage parameter alpha on dev_df.
        """
        if target_col is None:
            target_col = f"target_return_{self.horizon}d"

        y_train = train_df[target_col].values
        self.baseline_val = float(np.mean(y_train))

        # Compute residual targets
        y_residual_train = y_train - self.baseline_val

        valid_cols = [c for c in features if c in train_df.columns]
        X_train = np.nan_to_num(train_df[valid_cols].values, nan=0.0)

        # Subsample if large to keep memory under budget
        n_samples = min(len(X_train), 80000)
        np.random.seed(self.seed)
        sub_idx = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_tr_sub = X_train[sub_idx]
        y_res_sub = y_residual_train[sub_idx]

        # Initialize residual learner
        if self.residual_model_type == "ridge":
            self.residual_model = Ridge(alpha=200.0, random_state=self.seed).fit(X_tr_sub, y_res_sub)
        elif self.residual_model_type == "elastic_net":
            self.residual_model = ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=self.seed, max_iter=200).fit(X_tr_sub, y_res_sub)
        elif self.residual_model_type == "huber":
            self.residual_model = HuberRegressor(alpha=100.0, max_iter=200).fit(X_tr_sub, y_res_sub)
        elif self.residual_model_type == "catboost":
            self.residual_model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.04,
                depth=5,
                l2_leaf_reg=5.0,
                random_seed=self.seed,
                verbose=False,
                thread_count=1,
            ).fit(X_tr_sub, y_res_sub)
        else: # default: LightGBM
            self.residual_model = lgb.LGBMRegressor(
                objective="regression_l1",
                learning_rate=0.03,
                n_estimators=80,
                num_leaves=31,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=1,
                verbose=-1,
            ).fit(X_tr_sub, y_res_sub)

        # Tune shrinkage parameter alpha on dev_df if provided
        if dev_df is not None and len(dev_df) > 0:
            y_dev = dev_df[target_col].values
            X_dev = np.nan_to_num(dev_df[valid_cols].values, nan=0.0)
            raw_res_dev = self.residual_model.predict(X_dev)

            best_alpha = 0.0
            best_mae = float(np.mean(np.abs(y_dev - self.baseline_val))) # baseline loss

            for cand_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                cand_pred = self.baseline_val + cand_alpha * raw_res_dev
                cand_mae = float(np.mean(np.abs(y_dev - cand_pred)))
                if cand_mae < best_mae:
                    best_mae = cand_mae
                    best_alpha = cand_alpha

            self.alpha = best_alpha
        else:
            self.alpha = 0.5 # Default conservative shrinkage

    def predict(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Produces final return forecast: Baseline + alpha * ML residual
        """
        valid_cols = [c for c in features if c in df.columns]
        X = np.nan_to_num(df[valid_cols].values, nan=0.0)
        raw_res = self.residual_model.predict(X)
        final_ret = self.baseline_val + self.alpha * raw_res
        return final_ret

    def evaluate(self, val_df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Evaluates the residual learning model against matched validation observations.
        """
        target_col = f"target_return_{self.horizon}d"
        y_val = val_df[target_col].values
        val_close = val_df["close"].values
        preds = self.predict(val_df, features)
        metrics = compute_price_metrics(y_val, preds, val_close)
        return {
            "model_type": f"ResidualLearning_{self.residual_model_type.upper()}",
            "baseline_type": self.baseline_type,
            "learned_alpha": round(self.alpha, 2),
            **metrics,
        }
