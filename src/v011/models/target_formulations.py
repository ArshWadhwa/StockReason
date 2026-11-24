"""
StockReason v0.11 — Target Formulation Comparison Engine.
=========================================================
Remediates Phase 4 (Section 25):
Compares 5 mathematically rigorous target representations:
1. Simple Return: r = (P_{t+H} / P_t) - 1.0
2. Log Return: y = ln(P_{t+H} / P_t), reconstructed via P_{t+H} = P_t * exp(y + 0.5 * sigma^2)
3. Volatility-Normalized Return: z = r / sigma_{60d}, reconstructed via r = z * sigma_{60d}
4. Baseline Residual Return: eps = r - r_mean
5. Market-Adjusted Demeaned Return: r_tilde = r - r_nifty (reconstructed via causal market drift)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.v011.evaluation.baselines import compute_price_metrics


def compare_target_formulations_v011(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    horizon: int = 21,
    sample_train_limit: int = 60000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trains identical LightGBM architectures on 5 different target formulations,
    reconstructing nominal future price P_{t+H} consistently.
    """
    target_simple = f"target_return_{horizon}d"
    y_tr_simple = train_df[target_simple].values
    y_val_simple = val_df[target_simple].values
    val_close = val_df["close"].values
    n_eval = len(val_df)

    valid_cols = [c for c in features if c in train_df.columns]
    X_train = np.nan_to_num(train_df[valid_cols].values, nan=0.0)
    X_val = np.nan_to_num(val_df[valid_cols].values, nan=0.0)

    # Subsample for training efficiency
    np.random.seed(seed)
    sub_idx = np.random.choice(len(X_train), size=min(len(X_train), sample_train_limit), replace=False)
    X_tr_sub = X_train[sub_idx]
    y_tr_sub = y_tr_simple[sub_idx]

    results = []

    def fit_lgb(y_target):
        return lgb.LGBMRegressor(
            objective="regression_l1",
            learning_rate=0.03,
            n_estimators=80,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ).fit(X_tr_sub, y_target)

    # 1. Simple Return Target
    m_simple = fit_lgb(y_tr_sub)
    pred_simple = m_simple.predict(X_val)
    res_simple = compute_price_metrics(y_val_simple, pred_simple, val_close)
    results.append({"formulation": "1. Simple Return (r)", "loss_objective": "L1 Simple Return", **res_simple})

    # 2. Log Return Target
    y_tr_log = np.log(np.maximum(1.0 + y_tr_sub, 0.01))
    m_log = fit_lgb(y_tr_log)
    pred_log = m_log.predict(X_val)
    # Jensen's inequality correction: E[1 + r] = exp(mu_log + 0.5 * sigma^2)
    vol_proxy = (np.nan_to_num(val_df["vol_60d"].values, nan=0.25) * np.sqrt(horizon / 252.0)) ** 2
    pred_ret_from_log = np.exp(pred_log + 0.5 * vol_proxy) - 1.0
    res_log = compute_price_metrics(y_val_simple, pred_ret_from_log, val_close)
    results.append({"formulation": "2. Log Return ln(1+r) + Jensen", "loss_objective": "L1 Log Return", **res_log})

    # 3. Volatility-Normalized Return
    vol_tr = np.maximum(train_df["vol_60d"].values[sub_idx] * np.sqrt(horizon / 252.0), 0.02)
    vol_val = np.maximum(val_df["vol_60d"].values * np.sqrt(horizon / 252.0), 0.02)
    y_tr_vol_norm = y_tr_sub / vol_tr
    m_vol_norm = fit_lgb(y_tr_vol_norm)
    pred_z = m_vol_norm.predict(X_val)
    pred_ret_from_vol = pred_z * vol_val
    res_vol = compute_price_metrics(y_val_simple, pred_ret_from_vol, val_close)
    results.append({"formulation": "3. Vol-Normalized Return (r/sigma)", "loss_objective": "L1 Standardized Z", **res_vol})

    # 4. Baseline Residual Return
    mean_ret = float(np.mean(y_tr_sub))
    y_tr_res = y_tr_sub - mean_ret
    m_res = fit_lgb(y_tr_res)
    pred_res = m_res.predict(X_val)
    pred_ret_from_res = mean_ret + 0.5 * pred_res
    res_res = compute_price_metrics(y_val_simple, pred_ret_from_res, val_close)
    results.append({"formulation": "4. Baseline Residual Return", "loss_objective": "L1 Residual", **res_res})

    # 5. Market-Adjusted Demeaned Return
    mkt_ret_tr = train_df["nifty_ret_21d"].values[sub_idx] if "nifty_ret_21d" in train_df.columns else np.zeros(len(sub_idx))
    y_tr_mkt_adj = y_tr_sub - np.nan_to_num(mkt_ret_tr, nan=0.0)
    m_mkt = fit_lgb(y_tr_mkt_adj)
    pred_idio = m_mkt.predict(X_val)
    # Reconstruct with causal trailing market expectation (historical mean market return)
    hist_mkt_mean = float(np.mean(np.nan_to_num(mkt_ret_tr, nan=0.0)))
    pred_ret_from_mkt = hist_mkt_mean + pred_idio
    res_mkt = compute_price_metrics(y_val_simple, pred_ret_from_mkt, val_close)
    results.append({"formulation": "5. Market-Adjusted Return", "loss_objective": "L1 Demeaned", **res_mkt})

    return pd.DataFrame(results)
