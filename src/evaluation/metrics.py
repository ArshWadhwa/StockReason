"""Comprehensive Financial Forecasting Metrics Engine."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_forecasting_metrics(
    df: pd.DataFrame,
    pred_price_col: str,
    pred_ret_col: str,
    target_price_col: str = "target_price_21d",
    target_ret_col: str = "target_return_nominal_21d",
    close_col: str = "close",
    date_col: str = "date",
) -> Dict[str, Any]:
    """
    Computes rigorous practical and statistical forecasting metrics on matched valid cohort:
    - True Future-Price MAPE (%)
    - INR Price MAE (₹)
    - True Price MedAPE (%)
    - Price RMSE (₹)
    - Return MAE (%)
    - Return RMSE (%)
    - Directional Accuracy (%)
    - Mean Daily Rank IC (Spearman correlation)
    - Rank IC IR (Information Ratio)
    """
    valid = (
        df[target_price_col].notna()
        & df[pred_price_col].notna()
        & df[target_ret_col].notna()
        & df[pred_ret_col].notna()
        & df[close_col].notna()
        & (df[target_price_col] > 0)
    )
    sub = df[valid]
    n_obs = len(sub)
    if n_obs == 0:
        return {"n_obs": 0}

    y_true_p = sub[target_price_col].values
    y_pred_p = sub[pred_price_col].values
    y_true_r = sub[target_ret_col].values
    y_pred_r = sub[pred_ret_col].values

    # Price metrics
    price_abs_err = np.abs(y_true_p - y_pred_p)
    price_mae = float(np.mean(price_abs_err))
    price_rmse = float(np.sqrt(np.mean(price_abs_err ** 2)))
    price_pe = price_abs_err / np.maximum(y_true_p, 0.01)
    price_mape = float(np.mean(price_pe) * 100.0)
    price_medape = float(np.median(price_pe) * 100.0)

    # Return metrics
    ret_abs_err = np.abs(y_true_r - y_pred_r)
    ret_mae = float(np.mean(ret_abs_err) * 100.0)
    ret_rmse = float(np.sqrt(np.mean((y_true_r - y_pred_r) ** 2)) * 100.0)

    # Directional accuracy
    sign_true = np.sign(y_true_r)
    sign_pred = np.sign(y_pred_r)
    dir_acc = float(np.mean(sign_true == sign_pred) * 100.0)

    # Cross-sectional daily Rank IC
    mean_rank_ic = 0.0
    rank_ic_ir = 0.0
    if date_col in sub.columns and sub[date_col].nunique() > 1:
        daily_ics = []
        for _, grp_df in sub.groupby(date_col):
            if len(grp_df) >= 10:
                p_r = grp_df[pred_ret_col].values
                t_r = grp_df[target_ret_col].values
                if np.std(p_r) > 1e-8 and np.std(t_r) > 1e-8:
                    corr, _ = stats.spearmanr(p_r, t_r)
                    if not np.isnan(corr):
                        daily_ics.append(corr)

        if len(daily_ics) > 0:
            mean_rank_ic = float(np.mean(daily_ics))
            std_ic = float(np.std(daily_ics))
            rank_ic_ir = float(mean_rank_ic / (std_ic + 1e-8))

    return {
        "n_obs": n_obs,
        "true_price_mape": round(price_mape, 4),
        "inr_price_mae": round(price_mae, 4),
        "true_price_medape": round(price_medape, 4),
        "price_rmse": round(price_rmse, 4),
        "return_mae": round(ret_mae, 4),
        "return_rmse": round(ret_rmse, 4),
        "directional_accuracy": round(dir_acc, 2),
        "mean_rank_ic": round(mean_rank_ic, 4),
        "rank_ic_ir": round(rank_ic_ir, 4),
    }
