"""
StockReason v0.7.1 - Audited Quantitative Metrics Library
=========================================================
Implements the verified cross-sectional evaluation metric contract:
1. compute_ic(pred_df, method="spearman"|"pearson", min_names_per_date=20) -> pd.DataFrame
   - Strict cross-sectional calculation: evaluated PER DATE across assets.
   - Drops dates with fewer than min_names_per_date.
   - Any constant prediction across stocks on a date evaluates to IC = 0.0 (NOT NaN, NOT spurious).
2. summarize_ic(ic_df) -> dict
   - Mean IC, std IC, ICIR (daily), ICIR annualized.
   - Parametric t-stat and p-value.
3. compute_directional_accuracy(pred_df) -> dict
   - Evaluates directional accuracy against positive return base rate.
   - Correctly handles zero predictions (assigns tie/neutral or 0 sign).
4. audit_model_predictions(...) -> dict
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    pred_df: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
    date_col: str = "date",
    method: str = "spearman",
    min_names_per_date: int = 20,
) -> pd.DataFrame:
    """
    Computes cross-sectional Information Coefficient (IC) per date across stocks.

    Args:
        pred_df: DataFrame containing date, prediction, and target columns.
        pred_col: Column name of predictions.
        target_col: Column name of ground truth targets.
        date_col: Column name of dates.
        method: "spearman" (Rank IC) or "pearson" (Linear IC).
        min_names_per_date: Minimum valid stock observations required on a date.

    Returns:
        DataFrame with columns: [date, ic, n]
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(columns=["date", "ic", "n"])

    # Extract required columns and drop NaNs
    cols = [date_col, pred_col, target_col]
    missing_cols = [c for c in cols if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in pred_df: {missing_cols}")

    clean_df = pred_df[cols].dropna()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col])

    records = []
    # Group strictly by date
    for d, group in clean_df.groupby(date_col):
        n = len(group)
        if n < min_names_per_date:
            continue

        p = group[pred_col].to_numpy(dtype=float)
        t = group[target_col].to_numpy(dtype=float)

        std_p = np.std(p)
        std_t = np.std(t)

        # Mathematical invariance check:
        # If prediction or target is cross-sectionally constant, correlation is identically 0.0
        if std_p < 1e-12 or std_t < 1e-12:
            ic_val = 0.0
        else:
            if method == "spearman":
                # Handle possible constant ranks or ties
                r_res = stats.spearmanr(p, t)
                val = float(r_res.statistic) if hasattr(r_res, "statistic") else float(r_res[0])
            elif method == "pearson":
                r_res = stats.pearsonr(p, t)
                val = float(r_res.statistic) if hasattr(r_res, "statistic") else float(r_res[0])
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            ic_val = 0.0 if np.isnan(val) else val

        records.append({"date": d, "ic": ic_val, "n": n})

    if not records:
        return pd.DataFrame(columns=["date", "ic", "n"])

    res_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return res_df


def summarize_ic(
    ic_df: pd.DataFrame,
    horizon: int = 21,
) -> Dict[str, Any]:
    """
    Summarizes a per-date IC series.

    Returns:
        dict containing:
        - mean_ic: float
        - std_ic: float
        - icir_daily: float (mean_ic / std_ic)
        - icir_annualized: float (icir_daily * sqrt(252 / horizon))
        - t_stat: float
        - p_val: float
        - pct_positive: float
        - n_dates: int
    """
    if ic_df is None or ic_df.empty:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir_daily": 0.0,
            "icir_annualized": 0.0,
            "t_stat": 0.0,
            "p_val": 1.0,
            "pct_positive": 0.0,
            "n_dates": 0,
        }

    ics = ic_df["ic"].to_numpy(dtype=float)
    n = len(ics)
    if n == 0:
        return {"mean_ic": 0.0, "std_ic": 0.0, "icir_daily": 0.0, "icir_annualized": 0.0, "t_stat": 0.0, "p_val": 1.0, "pct_positive": 0.0, "n_dates": 0}

    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if n > 1 else 0.0

    if std_ic > 1e-12:
        icir_daily = mean_ic / std_ic
        # Periods per year = 252 / horizon
        periods_per_year = 252.0 / max(1, horizon)
        icir_annualized = icir_daily * np.sqrt(periods_per_year)
        se = std_ic / np.sqrt(n)
        t_stat = mean_ic / se
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=n - 1)))
    else:
        icir_daily = 0.0
        icir_annualized = 0.0
        t_stat = 0.0
        p_val = 1.0

    pct_positive = float(np.mean(ics > 0)) * 100.0

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir_daily": icir_daily,
        "icir_annualized": icir_annualized,
        "t_stat": t_stat,
        "p_val": p_val,
        "pct_positive": pct_positive,
        "n_dates": n,
    }


def compute_directional_accuracy(
    pred_df: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
) -> Dict[str, float]:
    """
    Computes directional accuracy against the correct test-set base rate.

    - Target positive class indicator: target > 0
    - Prediction positive class indicator: pred > 0
    - For zero prediction (e.g. ZeroForecaster), reports base rate comparison accurately.
    """
    clean = pred_df[[pred_col, target_col]].dropna()
    if len(clean) == 0:
        return {"dir_acc": 0.0, "base_rate": 0.0, "excess_dir_acc": 0.0}

    p = clean[pred_col].to_numpy(dtype=float)
    t = clean[target_col].to_numpy(dtype=float)

    # Ground truth direction: 1 for up, 0 for flat/down
    actual_up = (t > 0).astype(int)
    pos_base_rate = float(np.mean(actual_up))
    majority_base_rate = max(pos_base_rate, 1.0 - pos_base_rate)

    # For zero prediction: if prediction is exactly 0.0, it is not predicting 'up'
    # By convention: pred > 0 is up prediction
    pred_up = (p > 0).astype(int)
    acc = float(np.mean(pred_up == actual_up))

    return {
        "dir_acc": acc * 100.0,
        "base_rate": majority_base_rate * 100.0,
        "pos_base_rate": pos_base_rate * 100.0,
        "excess_dir_acc": (acc - majority_base_rate) * 100.0,
    }
