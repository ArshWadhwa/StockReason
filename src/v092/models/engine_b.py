"""
StockReason v0.9.2 — Engine B: Cross-Sectional Relative-Return & Ranking Engine.
================================================================================
Maintains independent relative-return ranking engine:
- Target: Cross-sectional demeaned forward return y_demeaned_{i, t}.
- Evaluates: Daily Spearman Rank IC, block-bootstrap 95% CI, non-overlapping IC,
  and residual IC (neutralized vs vol, size, beta, sector).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.v091.models.metrics import (
    compute_daily_cross_sectional_rank_ic,
    block_bootstrap_rank_ic_ci,
    compute_residual_rank_ic,
)

SEEDS = [42, 101, 202, 303, 404]


def evaluate_engine_b_ranking(
    X_train: np.ndarray,
    y_train_demeaned: np.ndarray,
    X_val: np.ndarray,
    y_val_demeaned: np.ndarray,
    val_df: pd.DataFrame,
    horizon: int = 21,
    date_col: str = "date_str",
) -> Dict[str, Any]:
    """
    Fits and evaluates cross-sectional ranking models.
    """
    results = []
    preds_dict = {}

    target_col = f"target_demeaned_{horizon}d" if f"target_demeaned_{horizon}d" in val_df.columns else "target_demeaned_21d"

    # Filter NaNs in training targets
    valid_tr = ~np.isnan(y_train_demeaned)
    X_tr_fit = X_train[valid_tr]
    y_tr_fit = y_train_demeaned[valid_tr]

    # 1. Ridge
    ridge = Ridge(alpha=100.0, random_state=42).fit(X_tr_fit, y_tr_fit)
    p_ridge = ridge.predict(X_val)
    preds_dict["Ridge"] = p_ridge

    # 2. LightGBM Ensemble
    lgb_preds = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(
            learning_rate=0.03,
            n_estimators=100,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_lambda=10.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )
        m.fit(X_tr_fit, y_tr_fit)
        lgb_preds.append(m.predict(X_val))
    p_lgb = np.mean(lgb_preds, axis=0)
    preds_dict["LightGBM"] = p_lgb

    # 3. Factor Momentum Baseline
    if "ret_252_21" in val_df.columns:
        p_mom = val_df["ret_252_21"].values
    elif "ret_126d" in val_df.columns:
        p_mom = val_df["ret_126d"].values
    else:
        p_mom = X_val[:, 5] # ret_126d
    preds_dict["Momentum_12_1"] = p_mom

    # Evaluate Rank IC & Residual IC for each
    for name, p in preds_dict.items():
        vdf = val_df[[date_col]].copy()
        vdf["pred"] = p
        vdf["target"] = y_val_demeaned
        
        ic_df = compute_daily_cross_sectional_rank_ic(vdf, "pred", "target", date_col=date_col)
        if not ic_df.empty:
            mean_ic, ic_low, ic_high = block_bootstrap_rank_ic_ci(ic_df["rank_ic"].values, block_size=21)
        else:
            mean_ic, ic_low, ic_high = 0.0, 0.0, 0.0

        # Residual IC
        val_df_res = val_df.copy()
        val_df_res["temp_pred"] = p
        res_ic_df = compute_residual_rank_ic(val_df_res, "temp_pred", target_col, date_col=date_col)
        if not res_ic_df.empty:
            mean_res_ic, res_low, res_high = block_bootstrap_rank_ic_ci(res_ic_df["residual_ic"].values, block_size=21)
        else:
            mean_res_ic, res_low, res_high = 0.0, 0.0, 0.0

        results.append({
            "model_or_factor": name,
            "rank_ic_mean": round(mean_ic, 4),
            "rank_ic_95_ci": f"[{ic_low:.4f}, {ic_high:.4f}]",
            "residual_ic_mean": round(mean_res_ic, 4),
            "residual_ic_95_ci": f"[{res_low:.4f}, {res_high:.4f}]",
        })

    return {
        "metrics_table": pd.DataFrame(results),
        "predictions": preds_dict,
    }
