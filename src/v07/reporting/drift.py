"""
StockReason v0.7 - Drift Monitor v2
====================================
Monitors production model health across three statistical dimensions:
  1. Performance Drift: Rolling cross-sectional Rank IC (alert if < 0.0 or drops significantly)
  2. Uncertainty Drift: Rolling empirical coverage of conformal prediction intervals
     (e.g., alert if 80% interval coverage drops below 70% or exceeds 95%)
  3. Feature Distribution Drift:
     - Population Stability Index (PSI): PSI >= 0.25 indicates significant shift
     - Kolmogorov-Smirnov (KS) 2-sample test: p-val < 0.01 indicates distributional shift
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.v07.reporting.metrics import compute_per_date_ic


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current samples.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Calculate quantile bin edges based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Ensure distinct bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Add small margins to outer edges
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts / len(ref)) + eps
    cur_pct = (cur_counts / len(cur)) + eps

    # Normalize back to 1
    ref_pct /= np.sum(ref_pct)
    cur_pct /= np.sum(cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))


def compute_rolling_rank_ic(
    pred_df: pd.DataFrame,
    window: int = 30,
    date_col: str = "date",
    pred_col: str = "y_pred",
    target_col: str = "y_true",
) -> pd.DataFrame:
    """
    Computes rolling mean Rank IC across dates.
    """
    per_date = compute_per_date_ic(pred_df, pred_col=pred_col, target_col=target_col, date_col=date_col)
    if per_date.empty:
        return pd.DataFrame(columns=["date", "rank_ic", "rolling_rank_ic"])

    per_date["rolling_rank_ic"] = per_date["rank_ic"].rolling(window=window, min_periods=max(5, window // 4)).mean()
    return per_date[["date", "rank_ic", "rolling_rank_ic"]]


def compute_rolling_coverage(
    df: pd.DataFrame,
    target_col: str = "y_true",
    lower_col: str = "interval_lower_80",
    upper_col: str = "interval_upper_80",
    window: int = 30,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Computes empirical coverage of conformal prediction intervals over a rolling window.
    """
    if lower_col not in df.columns or upper_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    valid = df[[date_col, target_col, lower_col, upper_col]].dropna().copy()
    valid["in_interval"] = (valid[target_col] >= valid[lower_col]) & (valid[target_col] <= valid[upper_col])

    # Aggregate daily
    daily = valid.groupby(date_col)["in_interval"].mean().reset_index()
    daily["rolling_coverage"] = daily["in_interval"].rolling(window=window, min_periods=max(5, window // 4)).mean()
    return daily


def check_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates PSI and KS test for key features.
    """
    results = {}
    for feat in features:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue

        ref_vals = ref_df[feat].dropna().to_numpy()
        cur_vals = cur_df[feat].dropna().to_numpy()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = calculate_psi(ref_vals, cur_vals)

        # KS test
        ks_res = stats.ks_2samp(ref_vals, cur_vals)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        if psi >= 0.25 or ks_pval < 0.001:
            level = "ALERT"
        elif psi >= 0.10 or ks_pval < 0.01:
            level = "WARNING"
        else:
            level = "HEALTHY"

        results[feat] = {
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 5),
            "drift_level": level,
        }

    return results


def check_drift(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    key_features: Optional[List[str]] = None,
    nominal_coverage: float = 0.80,
    min_rolling_rank_ic: float = 0.0,
    window: int = 30,
) -> Dict[str, Any]:
    """
    Executes full multi-dimensional drift audit.
    """
    alerts = []
    status = "HEALTHY"

    # 1. Performance Drift (Rolling Rank IC)
    p_col = "y_pred" if "y_pred" in current_df.columns else ("pred" if "pred" in current_df.columns else None)
    t_col = "y_true" if "y_true" in current_df.columns else ("actual" if "actual" in current_df.columns else None)
    d_col = "date" if "date" in current_df.columns else current_df.columns[0]

    rolling_ic_df = pd.DataFrame()
    latest_rolling_ic = None
    if p_col and t_col and d_col in current_df.columns:
        rolling_ic_df = compute_rolling_rank_ic(current_df, window=window, date_col=d_col, pred_col=p_col, target_col=t_col)
        if not rolling_ic_df.empty:
            valid_ics = rolling_ic_df["rolling_rank_ic"].dropna()
            if not valid_ics.empty:
                latest_rolling_ic = float(valid_ics.iloc[-1])
                if latest_rolling_ic < min_rolling_rank_ic:
                    alerts.append(
                        f"Rank IC Degradation: Rolling Rank IC is {latest_rolling_ic:.4f} (below threshold {min_rolling_rank_ic:.4f})"
                    )
                    status = "ALERT"

    # 2. Conformal Interval Coverage Drift
    low_col = "interval_lower_80" if "interval_lower_80" in current_df.columns else "lower_80"
    up_col = "interval_upper_80" if "interval_upper_80" in current_df.columns else "upper_80"

    rolling_cov_df = pd.DataFrame()
    latest_coverage = None
    if t_col and low_col in current_df.columns and up_col in current_df.columns:
        rolling_cov_df = compute_rolling_coverage(
            current_df, target_col=t_col, lower_col=low_col, upper_col=up_col, window=window, date_col=d_col
        )
        if not rolling_cov_df.empty:
            valid_covs = rolling_cov_df["rolling_coverage"].dropna()
            if not valid_covs.empty:
                latest_coverage = float(valid_covs.iloc[-1])
                # Nominal 80% coverage check
                if latest_coverage < (nominal_coverage - 0.10):
                    alerts.append(
                        f"Coverage Collapse: Conformal 80% interval observed coverage is {latest_coverage*100:.1f}% (< 70%)"
                    )
                    status = "ALERT"
                elif latest_coverage > 0.95:
                    alerts.append(
                        f"Over-Conservative Intervals: Conformal observed coverage is {latest_coverage*100:.1f}% (> 95%)"
                    )
                    if status != "ALERT":
                        status = "WARNING"

    # 3. Feature Distribution Drift
    feature_drift_res = {}
    if reference_df is not None and not reference_df.empty:
        feats_to_check = key_features or [
            c for c in current_df.columns
            if c not in [d_col, "ticker", "symbol", p_col, t_col, low_col, up_col, "regime", "sector"]
            and pd.api.types.is_numeric_dtype(current_df[c])
        ][:10]  # default top 10 numeric features

        feature_drift_res = check_feature_drift(reference_df, current_df, feats_to_check)
        for feat, info in feature_drift_res.items():
            if info["drift_level"] == "ALERT":
                alerts.append(f"Significant Feature Drift in '{feat}': PSI={info['psi']:.3f}, KS p-val={info['ks_pval']:.4f}")
                status = "ALERT"
            elif info["drift_level"] == "WARNING" and status != "ALERT":
                alerts.append(f"Moderate Feature Drift in '{feat}': PSI={info['psi']:.3f}")
                status = "WARNING"

    return {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_rolling_rank_ic": round(latest_rolling_ic, 4) if latest_rolling_ic is not None else None,
        "latest_coverage_80": round(latest_coverage, 4) if latest_coverage is not None else None,
        "alerts": alerts,
        "feature_drift": feature_drift_res,
        "num_alerts": len(alerts),
    }


def main():
    parser = argparse.ArgumentParser(description="StockReason v0.7 Drift Monitor v2")
    parser.add_argument("--window", type=int, default=30, help="Rolling window size in trading days")
    args = parser.parse_args()

    # Demonstration check
    res = check_drift(pd.DataFrame())
    print("\n[Drift Monitor Status]")
    print(f"  Status: {res['status']}")
    print(f"  Alerts: {len(res['alerts'])}")


if __name__ == "__main__":
    main()
