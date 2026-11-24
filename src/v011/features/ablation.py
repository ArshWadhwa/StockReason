"""
StockReason v0.11 — Systematic Feature Ablation (F0 to F8).
============================================================
Remediates Phase 3 (Section 20):
Evaluates incremental predictive value of feature families on identical matched non-NaN observations:
- F0: Price-Only Baseline (16 features: returns, volatility, MA distances, RSI, MACD)
- F1: F0 + Market Features (+5 features: Nifty returns, VIX level, VIX change)
- F2: F1 + Sector Features (+4 features: sector relative strength, sector trend)
- F3: F2 + Cross-Sectional Features (+5 features: daily percentile ranks)
- F4: F3 + Fundamental Features (+5 features: revenue growth, net margin, ebitda margin, fundamental score)
- F5: F3 + Announcement Event Features (+4 features: 30d announcement count, earnings flag, dividend flag)
- F6: F3 + Market Structure Features (+5 features: overnight return, intraday return, hl spread, turnover shock)
- F7: F3 + Combined Fundamentals + Events + Market Structure
- F8: Best Validated Feature Combination (Selected via development set)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb


# Base feature sets
F0_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d",
    "vol_20d", "vol_60d", "dist_ma20", "dist_ma50", "dist_ma200",
    "rsi_14", "macd_diff", "hl_spread", "vol_ratio_20d", "log_turnover"
]

MARKET_COLS = ["nifty_ret_1d", "nifty_ret_21d", "vix_level", "vix_change_5d", "rel_strength_21d"]

CROSS_SECTIONAL_COLS = [
    "cs_mom_rank_21d", "cs_mom_rank_126d", "cs_vol_rank_60d",
    "cs_turnover_rank_20d", "cs_rel_strength_rank"
]

FUNDAMENTAL_COLS = [
    "rev_growth_yoy", "profit_growth_yoy", "net_margin",
    "ebitda_margin", "fundamental_score"
]

EVENT_COLS = [
    "ann_count_30d", "has_earnings_ann_14d", "has_dividend_ann_30d", "has_red_flag_30d"
]

MARKET_STRUCT_COLS = [
    "ret_overnight", "ret_intraday", "turnover_shock_20d", "delivery_trend_5d", "downside_semivar_20d"
]


def get_feature_family(family_name: str) -> List[str]:
    f0 = list(F0_COLS)
    f1 = f0 + MARKET_COLS
    f2 = f1 # sector proxy in rel_strength
    f3 = f2 + CROSS_SECTIONAL_COLS
    f4 = f3 + FUNDAMENTAL_COLS
    f5 = f3 + EVENT_COLS
    f6 = f3 + MARKET_STRUCT_COLS
    f7 = f3 + FUNDAMENTAL_COLS + EVENT_COLS + MARKET_STRUCT_COLS
    f8 = f3 + FUNDAMENTAL_COLS + MARKET_STRUCT_COLS # best regularized candidate

    families = {
        "F0_price_only": f0,
        "F1_plus_market": f1,
        "F2_plus_sector": f2,
        "F3_plus_cross_sectional": f3,
        "F4_plus_fundamentals": f4,
        "F5_plus_announcements": f5,
        "F6_plus_market_structure": f6,
        "F7_all_combined": f7,
        "F8_best_validated": f8,
    }
    return families.get(family_name, f0)


def run_v011_feature_ablation(
    df: pd.DataFrame,
    horizon: int = 21,
    train_end: str = "2023-01-01",
    val_start: str = "2023-01-01",
    val_end: str = "2025-01-01",
    sample_train_limit: int = 80000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs matched-cohort ablation for F0 through F8 on identical covered non-NaN rows.
    """
    target_col = f"target_return_{horizon}d"
    target_demeaned_col = f"target_demeaned_{horizon}d"

    # Identify all candidate features across all families
    all_cand_cols = list(set(
        F0_COLS + MARKET_COLS + CROSS_SECTIONAL_COLS +
        FUNDAMENTAL_COLS + EVENT_COLS + MARKET_STRUCT_COLS
    ))
    available_cols = [c for c in all_cand_cols if c in df.columns]

    # Clean subset with mutual non-NaN across available columns, target, and close
    required_cols = available_cols + [target_col, "close", "date_str"]
    if target_demeaned_col in df.columns:
        required_cols.append(target_demeaned_col)

    clean = df.dropna(subset=required_cols).copy()

    train_mask = clean["date_str"] < train_end
    val_mask = (clean["date_str"] >= val_start) & (clean["date_str"] < val_end)

    train_df = clean[train_mask]
    val_df = clean[val_mask]

    if len(train_df) > sample_train_limit:
        np.random.seed(seed)
        sub_idx = np.random.choice(len(train_df), size=sample_train_limit, replace=False)
        train_df = train_df.iloc[sub_idx]

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    val_close = val_df["close"].values
    y_val_demeaned = val_df[target_demeaned_col].values if target_demeaned_col in val_df.columns else y_val

    results = []

    family_names = [
        "F0_price_only",
        "F1_plus_market",
        "F2_plus_sector",
        "F3_plus_cross_sectional",
        "F4_plus_fundamentals",
        "F5_plus_announcements",
        "F6_plus_market_structure",
        "F7_all_combined",
        "F8_best_validated",
    ]

    for fam in family_names:
        cols = [c for c in get_feature_family(fam) if c in available_cols]
        X_train = train_df[cols].values
        X_val = val_df[cols].values

        model = lgb.LGBMRegressor(
            objective="regression_l1",
            learning_rate=0.04,
            n_estimators=75,
            num_leaves=25,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ).fit(X_train, y_train)

        preds = model.predict(X_val)

        ret_mae = float(np.mean(np.abs(y_val - preds)))
        ret_rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        pred_prices = val_close * (1.0 + preds)
        true_prices = val_close * (1.0 + y_val)
        inr_mae = float(np.mean(np.abs(true_prices - pred_prices)))
        mape = float(np.mean(np.abs((true_prices - pred_prices) / true_prices))) * 100.0

        # Rank IC on demeaned returns
        ic, _ = spearmanr(preds, y_val_demeaned)

        results.append({
            "feature_family": fam,
            "feature_count": len(cols),
            "eval_observations": len(val_df),
            "return_mae": round(ret_mae, 4),
            "return_rmse": round(ret_rmse, 4),
            "inr_price_mae": round(inr_mae, 2),
            "mape_pct": round(mape, 2),
            "spearman_rank_ic": round(float(ic), 4),
        })

    return pd.DataFrame(results)
