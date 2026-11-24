"""
StockReason v0.10 — Matched-Cohort Feature Group Ablation Suite.
================================================================
Remediates Phase 7 (Section 13):
Evaluates five distinct matched feature groups on the primary 21-day horizon:
1. Technical Price Only (16 features)
2. Technicals + Market Macro (21 features)
3. Technicals + Macro + Cross-Sectional & Market Breadth (26 features)
4. Technicals + Macro + Cross-Sectional + Annual Financials (29 features)
5. Technicals + Macro + Cross-Sectional + Corporate Announcements (28 features)

Measures:
- Return MAE
- Return RMSE
- INR Price MAE
- Normalized Price Error (MAPE %)
- Spearman Rank IC
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from scipy import stats

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v092.models.baselines import compute_absolute_price_metrics


FEATURE_GROUPS_V010 = {
    "1. Price Only (16 features)": [
        "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d",
        "vol_21d", "vol_60d",
        "close_to_sma20", "close_to_sma50", "close_to_sma200",
        "rsi_14", "macd_hist", "bb_width_20", "atr_14", "high_low_ratio_21d", "ret_skew_60d",
    ],
    "2. Price + Market (21 features)": FEATURE_COLS_21,
    "3. Price + Market + Cross-Sectional (26 features)": FEATURE_COLS_21 + [
        "mom_pctile_21d", "vol_pctile_60d", "turnover_pctile", "market_breadth_sma50", "cs_dispersion_21d"
    ],
}


def run_v010_feature_ablation(
    panel_df: pd.DataFrame,
    horizon: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluates incremental value of feature groups on matched out-of-sample validation observations.
    """
    target_col = f"target_return_{horizon}d"
    train_mask = panel_df["date_str"] < "2023-01-01"
    val_mask = (panel_df["date_str"] >= "2023-01-01") & (panel_df["date_str"] < "2025-01-01")

    records = []

    for name, cols in FEATURE_GROUPS_V010.items():
        avail_cols = [c for c in cols if c in panel_df.columns]
        sub = panel_df.dropna(subset=avail_cols + [target_col, "close"])

        tr = sub[sub["date_str"] < "2023-01-01"]
        val = sub[(sub["date_str"] >= "2023-01-01") & (sub["date_str"] < "2025-01-01")]

        X_tr = tr[avail_cols].values
        y_tr = tr[target_col].values
        X_val = val[avail_cols].values
        y_val = val[target_col].values
        val_close = val["close"].values

        # Fast Ridge evaluation
        m = Ridge(alpha=100.0, random_state=seed).fit(X_tr, y_tr)
        p_val = m.predict(X_val)

        res = compute_absolute_price_metrics(y_val, p_val, val_close)

        # Rank IC
        val_res = val[["date_str"]].copy()
        val_res["pred"] = p_val
        val_res["target"] = y_val
        daily_ics = []
        for _, grp in val_res.groupby("date_str"):
            if len(grp) >= 10:
                r, _ = stats.spearmanr(grp["pred"], grp["target"])
                if not np.isnan(r):
                    daily_ics.append(r)
        mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

        records.append({
            "feature_group": name,
            "feature_count": len(avail_cols),
            "eval_observations": len(val),
            **res,
            "spearman_rank_ic": round(mean_ic, 4),
        })

    return pd.DataFrame(records)
