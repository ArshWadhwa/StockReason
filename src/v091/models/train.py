"""
StockReason v0.9.1 — Model Training & Walk-Forward IC Early Stopping.
=====================================================================
Remediates the Model Collapse Defect (Task 4 & Task 5):
1. Target: Cross-sectional demeaned forward return (target_demeaned_21d).
   Removes unconditional positive market drift (+1.94% to +3.61%).
2. Optimization: Chooses boosting rounds by walk-forward cross-sectional Rank IC (NOT MSE!).
3. Strong Regularization:
   - Learning rate: 0.02
   - Small trees: max_depth=4, num_leaves=15 to 31
   - Subsampling: row subsample=0.7, colsample=0.7
   - Minimum leaf samples: min_child_samples=50
   - L2 penalty: reg_lambda=10.0
   - Ensembles across 5 distinct random seeds.
4. Comprehensive Model & Baseline Comparison on Identical Rows:
   - Baselines: No-Change, Date Mean, Volatility Sort, Size Sort, Momentum, Reversal
   - Models: Ridge, LightGBM (5-seed ensemble), CatBoost (5-seed ensemble), PyTorch GRU
5. Evaluates:
   - Out-of-sample R^2 against per-date mean
   - Daily cross-sectional Rank IC & block-bootstrap 95% CI
   - Residual IC (neutralized vs vol, size, beta, sector)
   - Cross-sectional prediction dispersion
   - Mechanical Gate: if 95% CI includes 0 -> "No demonstrated edge".
"""

import os
import sys

# Critical on macOS: OpenMP and thread settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn as nn

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.v091.features.pipeline import FEATURE_COLS_21, StockReasonPipelineV091
from src.v091.models.metrics import (
    evaluate_institutional_metrics,
    compute_daily_cross_sectional_rank_ic,
    block_bootstrap_rank_ic_ci
)
from src.v091.models.baselines import compute_factor_signals, generate_matched_random_baskets

SEEDS = [42, 101, 202, 303, 404]


class CustomRankICEarlyStopping:
    """Callback for LightGBM to perform early stopping strictly on cross-sectional Rank IC."""
    def __init__(self, val_df: pd.DataFrame, target_col: str, date_col: str = "date_str", patience: int = 25):
        self.val_df = val_df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.patience = patience
        self.best_ic = -1.0
        self.best_iteration = 0
        self.ic_history = []
        self.no_improve = 0

    def __call__(self, env):
        # Predict on validation features
        model = env.model
        preds = model.predict(self.val_df[FEATURE_COLS_21].values)
        
        self.val_df["temp_pred"] = preds
        df_ic = compute_daily_cross_sectional_rank_ic(self.val_df, "temp_pred", self.target_col, date_col=self.date_col)
        curr_ic = float(df_ic["rank_ic"].mean()) if not df_ic.empty else 0.0
        self.ic_history.append(curr_ic)
        
        if curr_ic > self.best_ic + 1e-5:
            self.best_ic = curr_ic
            self.best_iteration = env.iteration
            self.no_improve = 0
        else:
            self.no_improve += 1
            
        if self.no_improve >= self.patience:
            env.model.best_iteration = self.best_iteration
            raise EarlyStopException(env.iteration, self.best_iteration, self.best_ic)


class EarlyStopException(Exception):
    def __init__(self, iteration, best_iteration, best_score):
        self.iteration = iteration
        self.best_iteration = best_iteration
        self.best_score = best_score


def train_lightgbm_rank_ic_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dates: List[str],
    seeds: List[int] = SEEDS,
) -> Tuple[List[Any], List[int], np.ndarray]:
    """
    Trains an ensemble of 5 LightGBM models with Rank IC early stopping.
    Fits once per seed with n_estimators=150, then selects optimal tree cutoff by validation Rank IC.
    """
    val_records = pd.DataFrame({"date_str": val_dates, "target": y_val})
    
    models = []
    best_iters = []
    val_preds_list = []
    
    candidate_trees = [10, 20, 30, 40, 50, 75, 100, 125, 150]
    
    for s in seeds:
        clf = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            random_state=s,
            n_jobs=4,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        
        best_seed_ic = -999.0
        best_iter = 150
        
        for n_trees in candidate_trees:
            preds_v = clf.predict(X_val, num_iteration=n_trees)
            val_records["pred"] = preds_v
            
            df_ic = compute_daily_cross_sectional_rank_ic(val_records, "pred", "target")
            ic_val = float(df_ic["rank_ic"].mean()) if not df_ic.empty else 0.0
            
            if ic_val > best_seed_ic:
                best_seed_ic = ic_val
                best_iter = n_trees
                
        models.append(clf)
        best_iters.append(best_iter)
        val_preds_list.append(clf.predict(X_val, num_iteration=best_iter))
        
    avg_val_preds = np.mean(val_preds_list, axis=0)
    return models, best_iters, avg_val_preds


def train_catboost_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dates: List[str],
    seeds: List[int] = SEEDS,
) -> Tuple[List[Any], List[int], np.ndarray]:
    """Trains 5-seed CatBoost ensemble, fitting once per seed and selecting optimal iteration."""
    val_records = pd.DataFrame({"date_str": val_dates, "target": y_val})
    models = []
    best_iters = []
    val_preds_list = []
    
    candidate_iters = [10, 25, 50, 75, 100]
    
    for s in seeds:
        cb = CatBoostRegressor(
            iterations=100,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=10.0,
            random_seed=s,
            verbose=0,
            thread_count=4
        )
        cb.fit(X_train, y_train)
        
        best_ic = -999.0
        best_iter = 100
        
        for n_iter in candidate_iters:
            p = cb.predict(X_val, ntree_end=n_iter)
            val_records["pred"] = p
            df_ic = compute_daily_cross_sectional_rank_ic(val_records, "pred", "target")
            ic_val = float(df_ic["rank_ic"].mean()) if not df_ic.empty else 0.0
            if ic_val > best_ic:
                best_ic = ic_val
                best_iter = n_iter
                
        models.append(cb)
        best_iters.append(best_iter)
        val_preds_list.append(cb.predict(X_val, ntree_end=best_iter))
        
    avg_val_preds = np.mean(val_preds_list, axis=0)
    return models, best_iters, avg_val_preds
