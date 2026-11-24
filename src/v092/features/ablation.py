"""
StockReason v0.9.2 — Matched-Cohort Ablation Engine for Absolute Price Forecasting.
=====================================================================================
Evaluates feature groups on identical rows and dates for absolute price forecasting:
- Group 1: Price Only (16 features)
- Group 2: Price + Market (21 features)
- Group 3: Price + Market + Sector (23 features)
- Group 4: Price + Market + Sector + Annual Financials (matched cohort)
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v092.features.external_data import load_annual_financial_ratios, load_sector_indices_features


def run_absolute_price_matched_ablation(
    panel_df: pd.DataFrame,
    horizon: int = 21,
    train_end_date: str = "2022-12-31",
    val_end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Executes matched-cohort ablation for absolute return and price forecasting.
    Reports: Return MAE, Return RMSE, INR Price MAE, Normalized Price Error (%), and Rank IC.
    """
    df = panel_df.copy()
    if "date_str" not in df.columns:
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    target_ret_col = f"target_return_{horizon}d"
    target_price_col = f"target_price_{horizon}d"

    # Merge annual financials with strict publication lag
    fin_df = load_annual_financial_ratios()
    if not fin_df.empty:
        df["dt"] = pd.to_datetime(df["date_str"])
        fin_df["dt"] = pd.to_datetime(fin_df["available_from_date"])
        df = df.sort_values(["symbol", "dt"])
        fin_df = fin_df.sort_values(["symbol", "dt"])
        
        merged_parts = []
        for sym, grp in df.groupby("symbol"):
            sym_fin = fin_df[fin_df["symbol"] == sym]
            if not sym_fin.empty:
                m = pd.merge_asof(
                    grp,
                    sym_fin[["dt", "profit_margin", "roe", "asset_turnover"]],
                    on="dt",
                    direction="backward"
                )
                merged_parts.append(m)
            else:
                grp["profit_margin"] = np.nan
                grp["roe"] = np.nan
                grp["asset_turnover"] = np.nan
                merged_parts.append(grp)
        df = pd.concat(merged_parts, ignore_index=True)
        if "dt" in df.columns:
            df = df.drop(columns=["dt"])
    else:
        df["profit_margin"] = np.nan
        df["roe"] = np.nan
        df["asset_turnover"] = np.nan

    # Add sector return feature
    sector_dfs = load_sector_indices_features()
    if "NIFTY_500" in sector_dfs:
        n500 = sector_dfs["NIFTY_500"]
        df = df.merge(n500[["date_str", "nifty_500_ret_21d"]], on="date_str", how="left")
    else:
        df["nifty_500_ret_21d"] = df["nifty_ret_21d"] if "nifty_ret_21d" in df.columns else 0.0

    # Define feature groups
    price_16 = [
        "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d",
        "vol_20d", "vol_60d",
        "dist_ma20", "dist_ma50", "dist_ma200",
        "rsi_14", "macd_diff", "hl_spread", "vol_ratio_20d", "log_turnover"
    ]
    price_market_21 = FEATURE_COLS_21
    price_market_sector_22 = price_market_21 + ["nifty_500_ret_21d"]
    all_with_fin = price_market_sector_22 + ["profit_margin", "roe", "asset_turnover"]

    groups = [
        ("1. Price Only (16 features)", price_16, False),
        ("2. Price + Market (21 features)", price_market_21, False),
        ("3. Price + Market + Sector (22 features)", price_market_sector_22, False),
        ("4. Fundamentals Cohort (25 features)", all_with_fin, True),
    ]

    # Matched evaluation cohort: validation set 2023-2024
    results = []

    for name, f_list, requires_fin in groups:
        sub_df = df.dropna(subset=f_list + [target_ret_col, "close"])
        if requires_fin:
            sub_df = sub_df.dropna(subset=["profit_margin", "roe"])

        train_data = sub_df[sub_df["date_str"] <= train_end_date]
        val_data = sub_df[(sub_df["date_str"] > train_end_date) & (sub_df["date_str"] <= val_end_date)]

        if len(train_data) < 100 or len(val_data) < 100:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_data[f_list].values)
        y_tr = train_data[target_ret_col].values

        X_val = scaler.transform(val_data[f_list].values)
        y_val = val_data[target_ret_col].values
        val_close = val_data["close"].values
        true_price = val_close * (1.0 + y_val)

        model = Ridge(alpha=100.0, random_state=42).fit(X_tr, y_tr)
        pred_ret = model.predict(X_val)
        pred_price = val_close * (1.0 + pred_ret)

        ret_mae = float(np.mean(np.abs(y_val - pred_ret)))
        ret_rmse = float(np.sqrt(np.mean((y_val - pred_ret) ** 2)))
        inr_mae = float(np.mean(np.abs(true_price - pred_price)))
        mape_pct = float(np.mean(np.abs(true_price - pred_price) / (true_price + 1e-4)) * 100.0)

        # Cross-sectional Rank IC
        val_df_ic = val_data[["date_str"]].copy()
        val_df_ic["pred"] = pred_ret
        val_df_ic["target"] = y_val
        ics = []
        for _, g in val_df_ic.groupby("date_str"):
            if len(g) >= 10:
                r, _ = stats.spearmanr(g["pred"], g["target"])
                if not np.isnan(r):
                    ics.append(r)
        mean_ic = float(np.mean(ics)) if ics else 0.0

        results.append({
            "feature_group": name,
            "feature_count": len(f_list),
            "eval_observations": len(val_data),
            "return_mae": round(ret_mae, 4),
            "return_rmse": round(ret_rmse, 4),
            "inr_price_mae": round(inr_mae, 2),
            "normalized_price_error_pct": round(mape_pct, 2),
            "rank_ic": round(mean_ic, 4),
        })

    return pd.DataFrame(results)
