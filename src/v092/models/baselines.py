"""
StockReason v0.9.2 — Strict Baseline Suite for Absolute Price Forecasting (Engine A).
=====================================================================================
Remediates Item 4:
- Evaluates competitive baselines fitted strictly on eligible training data:
  1. No-Change Baseline: r_hat = 0.0 => P_hat_{t+H} = P_t
  2. Historical Mean Baseline: r_hat = mean(r_train)
  3. Historical Median Baseline: r_hat = median(r_train)
  4. Sector-Return Baseline: r_hat = mean(r_sector_train)
  5. Market-Aware Baseline: r_hat_i = alpha_i + beta_i * r_market_hat
- Computes:
  - Return MAE
  - Return RMSE
  - INR Price MAE
  - Normalized Price Error (%)
  - Directional Accuracy (%)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_absolute_price_metrics(
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
    current_prices: np.ndarray,
) -> Dict[str, float]:
    """
    Computes absolute price forecasting performance metrics.
    """
    true_price = current_prices * (1.0 + y_true_ret)
    pred_price = current_prices * (1.0 + y_pred_ret)

    ret_mae = float(np.mean(np.abs(y_true_ret - y_pred_ret)))
    ret_rmse = float(np.sqrt(np.mean((y_true_ret - y_pred_ret) ** 2)))

    price_diff = np.abs(true_price - pred_price)
    inr_mae = float(np.mean(price_diff))
    mape_pct = float(np.mean(price_diff / (true_price + 1e-4)) * 100.0)

    # Directional accuracy
    dir_true = (y_true_ret > 0).astype(int)
    dir_pred = (y_pred_ret > 0).astype(int)
    acc_pct = float(np.mean(dir_true == dir_pred) * 100.0)

    return {
        "return_mae": round(ret_mae, 4),
        "return_rmse": round(ret_rmse, 4),
        "inr_price_mae": round(inr_mae, 2),
        "normalized_price_error_pct": round(mape_pct, 2),
        "directional_accuracy_pct": round(acc_pct, 2),
    }


def evaluate_engine_a_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """
    Fits and evaluates the 5 strict baselines on validation data.
    """
    target_col = f"target_return_{horizon}d"
    
    clean_tr = train_df.dropna(subset=[target_col, "close"])
    clean_val = val_df.dropna(subset=[target_col, "close"])

    y_tr = clean_tr[target_col].values
    y_val = clean_val[target_col].values
    val_close = clean_val["close"].values

    # 1. No-Change Baseline (r_hat = 0.0)
    pred_nochange = np.zeros(len(y_val))
    m_nochange = compute_absolute_price_metrics(y_val, pred_nochange, val_close)

    # 2. Historical Mean Baseline
    mean_ret = float(np.mean(y_tr))
    pred_mean = np.full(len(y_val), mean_ret)
    m_mean = compute_absolute_price_metrics(y_val, pred_mean, val_close)

    # 3. Historical Median Baseline
    median_ret = float(np.median(y_tr))
    pred_median = np.full(len(y_val), median_ret)
    m_median = compute_absolute_price_metrics(y_val, pred_median, val_close)

    # 4. Sector-Return Baseline
    # Group by sector if available, otherwise fallback to market mean
    if "sector" in clean_tr.columns and "sector" in clean_val.columns:
        sector_means = clean_tr.groupby("sector")[target_col].mean().to_dict()
        pred_sector = clean_val["sector"].map(sector_means).fillna(mean_ret).values
    else:
        pred_sector = pred_mean
    m_sector = compute_absolute_price_metrics(y_val, pred_sector, val_close)

    # 5. Market-Aware Baseline (CAPM beta * nifty_ret)
    mkt_col = "nifty_ret_21d" if "nifty_ret_21d" in clean_val.columns else "nifty_ret_1d"
    if mkt_col in clean_tr.columns and mkt_col in clean_val.columns:
        # Simple cross-sectional regression beta
        cov = np.cov(clean_tr[target_col].values, clean_tr[mkt_col].values)[0, 1]
        var_m = np.var(clean_tr[mkt_col].values) + 1e-6
        beta = float(cov / var_m)
        alpha = mean_ret - beta * float(np.mean(clean_tr[mkt_col].values))
        pred_market = alpha + beta * clean_val[mkt_col].values
    else:
        pred_market = pred_mean
    m_market = compute_absolute_price_metrics(y_val, pred_market, val_close)

    baselines_records = [
        {"model_or_baseline": "Baseline: No-Change (P_hat = P_t)", **m_nochange},
        {"model_or_baseline": "Baseline: Historical Mean", **m_mean},
        {"model_or_baseline": "Baseline: Historical Median", **m_median},
        {"model_or_baseline": "Baseline: Sector Return", **m_sector},
        {"model_or_baseline": "Baseline: Market-Aware CAPM", **m_market},
    ]
    return pd.DataFrame(baselines_records)
