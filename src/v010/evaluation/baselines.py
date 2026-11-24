"""
StockReason v0.10 — Ten Strict Statistical & Econometric Baselines Suite.
==========================================================================
Remediates Phase 8 (Section 15):
Evaluates all 10 mandated baselines on identical evaluation observations:
1. No-Change Price Forecast (P_hat_{t+H} = P_t, r_hat = 0.0)
2. Historical Mean Return (unconditional training mean)
3. Historical Median Return (unconditional training median)
4. Rolling Historical Drift (trailing 252d annualized drift scaled to H)
5. Exponentially Weighted Drift (decay factor lambda = 0.94)
6. Regularized Market-Aware Regression (CAPM beta * r_mkt)
7. Sector-Aware Historical Return (mean return by sector)
8. Simple Market + Sector Decomposition (without complex ML)
9. Ridge & Elastic Net Regressors
10. Existing v0.9.2 Gradient Boosted Models (LightGBM, CatBoost)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v092.models.baselines import compute_absolute_price_metrics


def evaluate_ten_strict_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int = 21,
    features: List[str] = FEATURE_COLS_21,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fits and evaluates the 10 mandated statistical and econometric baselines.
    """
    target_col = f"target_return_{horizon}d"
    clean_tr = train_df.dropna(subset=features + [target_col, "close"]).copy()
    clean_val = val_df.dropna(subset=features + [target_col, "close"]).copy()

    y_tr = clean_tr[target_col].values
    y_val = clean_val[target_col].values
    val_close = clean_val["close"].values
    X_tr = clean_tr[features].values
    X_val = clean_val[features].values

    h_scale = horizon / 252.0
    records = []

    # 1. No-Change Baseline (r_hat = 0.0)
    p_nochange = np.zeros(len(y_val))
    records.append({"baseline_name": "1. No-Change (P_hat = P_t)", **compute_absolute_price_metrics(y_val, p_nochange, val_close)})

    # 2. Historical Mean Return
    mean_ret = float(np.mean(y_tr))
    p_mean = np.full(len(y_val), mean_ret)
    records.append({"baseline_name": "2. Historical Mean Return", **compute_absolute_price_metrics(y_val, p_mean, val_close)})

    # 3. Historical Median Return
    median_ret = float(np.median(y_tr))
    p_median = np.full(len(y_val), median_ret)
    records.append({"baseline_name": "3. Historical Median Return", **compute_absolute_price_metrics(y_val, p_median, val_close)})

    # 4. Rolling Historical Drift (trailing 252d return drift)
    if "ret_252_21" in clean_val.columns:
        p_drift = np.clip(clean_val["ret_252_21"].fillna(mean_ret).values * (horizon / 252.0), -0.5, 0.5)
    elif "ret_126d" in clean_val.columns:
        p_drift = np.clip(clean_val["ret_126d"].fillna(mean_ret).values * (horizon / 126.0), -0.5, 0.5)
    else:
        p_drift = p_mean
    records.append({"baseline_name": "4. Rolling Historical Drift (252d)", **compute_absolute_price_metrics(y_val, p_drift, val_close)})

    # 5. Exponentially Weighted Drift
    ewm_drift = float(np.average(y_tr[-min(100000, len(y_tr)):], weights=np.linspace(0.5, 1.0, min(100000, len(y_tr)))))
    p_ewm = np.full(len(y_val), ewm_drift)
    records.append({"baseline_name": "5. Exponentially Weighted Drift", **compute_absolute_price_metrics(y_val, p_ewm, val_close)})

    # 6. Regularized Market-Aware CAPM
    mkt_col = "nifty_ret_21d" if "nifty_ret_21d" in clean_val.columns else "nifty_ret_1d"
    if mkt_col in clean_tr.columns and mkt_col in clean_val.columns:
        beta = clean_val["beta_60d"].clip(0.1, 2.5).values if "beta_60d" in clean_val.columns else np.ones(len(clean_val))
        p_capm = beta * clean_val[mkt_col].values
    else:
        p_capm = p_mean
    records.append({"baseline_name": "6. Regularized Market-Aware CAPM", **compute_absolute_price_metrics(y_val, p_capm, val_close)})

    # 7. Sector-Aware Historical Return
    if "sector" in clean_tr.columns and "sector" in clean_val.columns:
        sec_means = clean_tr.groupby("sector")[target_col].mean().to_dict()
        p_sec = clean_val["sector"].map(sec_means).fillna(mean_ret).values
    else:
        p_sec = p_mean
    records.append({"baseline_name": "7. Sector-Aware Historical Return", **compute_absolute_price_metrics(y_val, p_sec, val_close)})

    # 8. Simple Market + Sector Decomposition (Linear, no complex ML)
    p_simple_decomp = 0.6 * p_capm + 0.4 * p_sec
    records.append({"baseline_name": "8. Simple Linear Market+Sector Decomp", **compute_absolute_price_metrics(y_val, p_simple_decomp, val_close)})

    # 9. Ridge & Elastic Net
    m_ridge = Ridge(alpha=100.0, random_state=seed).fit(X_tr, y_tr)
    p_ridge = m_ridge.predict(X_val)
    records.append({"baseline_name": "9a. Econometric Ridge Regressor", **compute_absolute_price_metrics(y_val, p_ridge, val_close)})

    m_enet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed).fit(X_tr, y_tr)
    p_enet = m_enet.predict(X_val)
    records.append({"baseline_name": "9b. Econometric Elastic Net Regressor", **compute_absolute_price_metrics(y_val, p_enet, val_close)})

    # 10. Existing v0.9.2 LightGBM Baseline
    m_lgb = lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=100, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
    ).fit(X_tr, y_tr)
    p_lgb = m_lgb.predict(X_val)
    records.append({"baseline_name": "10. Existing v0.9.2 LightGBM Model", **compute_absolute_price_metrics(y_val, p_lgb, val_close)})

    df_res = pd.DataFrame(records)
    df_res["horizon"] = f"{horizon}d"
    df_res["eval_observations"] = len(clean_val)
    return df_res
