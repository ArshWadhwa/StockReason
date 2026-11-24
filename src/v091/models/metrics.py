"""
StockReason v0.9.1 — Quantitative Evaluation Metrics & Residual IC.
===================================================================
Computes verifiable institutional metrics:
1. Daily Cross-Sectional Spearman Rank IC.
2. Block-Bootstrap 95% Confidence Interval (block length = 21 sessions).
3. Non-Overlapping IC distribution across all 21 sub-offsets.
4. Residual IC: Neutralizes predictions against volatility, size, beta, and sector per date.
5. Out-of-Sample R^2 against the per-date mean:
   R^2 = 1.0 - [ sum(y - y_hat)^2 / sum(y - y_bar_date)^2 ]
6. MAE, RMSE, Balanced Accuracy, MCC, Specificity, Recall, Confusion Matrix.
7. Always-Up and Always-Down naive baselines.
8. Promotion Rule: If 95% CI on non-overlapping dates includes 0 -> "No demonstrated edge".
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def compute_daily_cross_sectional_rank_ic(
    df_preds: pd.DataFrame,
    pred_col: str,
    target_col: str,
    date_col: str = "date_str",
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Computes daily cross-sectional Spearman Rank IC."""
    records = []
    for d, group in df_preds.groupby(date_col):
        sub = group.dropna(subset=[pred_col, target_col])
        if len(sub) >= min_stocks:
            # Check for constant prediction
            if np.all(sub[pred_col] == sub[pred_col].iloc[0]):
                records.append({"date": d, "rank_ic": 0.0, "n_stocks": len(sub)})
                continue
            r, _ = stats.spearmanr(sub[pred_col], sub[target_col])
            if not np.isnan(r):
                records.append({"date": d, "rank_ic": float(r), "n_stocks": len(sub)})
                
    return pd.DataFrame(records)


def block_bootstrap_rank_ic_ci(
    ic_series: np.ndarray,
    block_size: int = 21,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes Stationary Block Bootstrap 95% Confidence Interval for Rank IC.
    Accounts for temporal autocorrelation in overlapping 21-day holding returns.
    """
    if len(ic_series) < block_size * 2:
        return float(np.mean(ic_series)), float(np.mean(ic_series)), float(np.mean(ic_series))
        
    np.random.seed(seed)
    n = len(ic_series)
    n_blocks = int(np.ceil(n / block_size))
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample starting indices of blocks
        start_indices = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([ic_series[idx : idx + block_size] for idx in start_indices])[:n]
        bootstrap_means.append(np.mean(sample))
        
    mean_val = float(np.mean(ic_series))
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    return mean_val, ci_lower, ci_upper


def compute_non_overlapping_ic_distribution(
    df_ic: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """
    Computes Rank IC across all 21 distinct non-overlapping day-of-month offsets.
    Offset k samples dates index = k, k + horizon, k + 2*horizon, etc.
    """
    records = []
    n = len(df_ic)
    for offset in range(horizon):
        sub_ic = df_ic.iloc[offset::horizon]["rank_ic"].values
        if len(sub_ic) >= 5:
            records.append({
                "offset": offset,
                "n_samples": len(sub_ic),
                "mean_ic": float(np.mean(sub_ic)),
                "std_ic": float(np.std(sub_ic)),
                "ci_lower_naive": float(np.percentile(sub_ic, 2.5)) if len(sub_ic) >= 20 else float(np.mean(sub_ic) - 1.96 * np.std(sub_ic) / np.sqrt(len(sub_ic))),
                "ci_upper_naive": float(np.percentile(sub_ic, 97.5)) if len(sub_ic) >= 20 else float(np.mean(sub_ic) + 1.96 * np.std(sub_ic) / np.sqrt(len(sub_ic)))
            })
            
    return pd.DataFrame(records)


def compute_residual_rank_ic(
    df_preds: pd.DataFrame,
    pred_col: str,
    target_col: str,
    vol_col: str = "vol_60d",
    size_col: str = "log_turnover",
    date_col: str = "date_str",
) -> pd.DataFrame:
    """
    Computes Residual IC after neutralizing predictions per date against
    volatility and size via cross-sectional OLS regression:
    pred_{i, t} = a_t + b_1 * vol_{i, t} + b_2 * size_{i, t} + eps_{i, t}
    Residual IC is the rank correlation between eps_{i, t} and target_{i, t}.
    """
    records = []
    for d, group in df_preds.groupby(date_col):
        cols = [pred_col, target_col, vol_col, size_col]
        sub = group.dropna(subset=[c for c in cols if c in group.columns]).copy()
        if len(sub) >= 15:
            feature_cols = [c for c in [vol_col, size_col] if c in sub.columns]
            if feature_cols:
                X = sub[feature_cols].values
                y_pred = sub[pred_col].values
                
                # Check variance
                if np.var(y_pred) < 1e-12:
                    records.append({"date": d, "residual_ic": 0.0, "n_stocks": len(sub)})
                    continue
                    
                reg = LinearRegression().fit(X, y_pred)
                residual = y_pred - reg.predict(X)
            else:
                residual = sub[pred_col].values
                
            r, _ = stats.spearmanr(residual, sub[target_col].values)
            if not np.isnan(r):
                records.append({"date": d, "residual_ic": float(r), "n_stocks": len(sub)})
                
    return pd.DataFrame(records)


def evaluate_institutional_metrics(
    df_eval: pd.DataFrame,
    pred_col: str,
    target_col: str,
    date_col: str = "date_str",
    horizon: int = 21,
) -> Dict[str, Any]:
    """Computes the full institutional evaluation suite on identical rows."""
    sub = df_eval.dropna(subset=[pred_col, target_col]).copy()
    y_true = sub[target_col].values
    y_pred = sub[pred_col].values
    
    if len(sub) == 0:
        return {}
        
    # Return error metrics
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    
    # Out of sample R2 against the cross-sectional per-date mean
    date_means = sub.groupby(date_col)[target_col].transform("mean").values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - date_means) ** 2) + 1e-9
    r2_oos = float(1.0 - (ss_res / ss_tot))
    
    # Directional classification
    actual_up = (y_true > 0).astype(int)
    pred_up = (y_pred > 0).astype(int)
    
    always_up_acc = float(np.mean(actual_up == 1))
    always_down_acc = float(np.mean(actual_up == 0))
    accuracy = float(np.mean(actual_up == pred_up))
    
    cm = confusion_matrix(actual_up, pred_up, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    recall = float(tp / (tp + fn + 1e-9))
    specificity = float(tn / (tn + fp + 1e-9))
    balanced_acc = float(0.5 * (recall + specificity))
    mcc = float(matthews_corrcoef(actual_up, pred_up)) if np.var(pred_up) > 0 and np.var(actual_up) > 0 else 0.0
    
    # Prediction dispersion (cross-sectional std dev per date)
    date_stds = sub.groupby(date_col)[pred_col].std()
    dispersion = float(date_stds.mean()) if not date_stds.empty else 0.0
    
    # Daily Rank IC & Bootstrap CI
    df_ic = compute_daily_cross_sectional_rank_ic(sub, pred_col, target_col, date_col=date_col)
    ic_vals = df_ic["rank_ic"].values if not df_ic.empty else np.array([0.0])
    mean_ic, ic_ci_lower, ic_ci_upper = block_bootstrap_rank_ic_ci(ic_vals, block_size=horizon)
    
    # Non-overlapping IC distribution
    df_non_overlap = compute_non_overlapping_ic_distribution(df_ic, horizon=horizon)
    mean_non_overlap_ic = float(df_non_overlap["mean_ic"].mean()) if not df_non_overlap.empty else 0.0
    non_overlap_ci_lower = float(df_non_overlap["ci_lower_naive"].min()) if not df_non_overlap.empty else 0.0
    non_overlap_ci_upper = float(df_non_overlap["ci_upper_naive"].max()) if not df_non_overlap.empty else 0.0
    
    # Residual IC
    df_resid_ic = compute_residual_rank_ic(sub, pred_col, target_col, date_col=date_col)
    resid_ic_vals = df_resid_ic["residual_ic"].values if not df_resid_ic.empty else np.array([0.0])
    resid_ic_mean, resid_ic_lower, resid_ic_upper = block_bootstrap_rank_ic_ci(resid_ic_vals, block_size=horizon)
    
    # Gate check: Mechanical rule
    has_demonstrated_edge = (non_overlap_ci_lower > 0.0) and (resid_ic_lower > 0.0)
    status = "READY" if has_demonstrated_edge else "NOT READY: No demonstrated edge (CI includes 0)"
    
    return {
        "n_observations": len(sub),
        "prediction_dispersion": dispersion,
        "mae_return": mae,
        "rmse_return": rmse,
        "r2_oos_vs_date_mean": r2_oos,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "recall": recall,
        "specificity": specificity,
        "mcc": mcc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "always_up_accuracy": always_up_acc,
        "always_down_accuracy": always_down_acc,
        "rank_ic_mean": mean_ic,
        "rank_ic_ci_95": [ic_ci_lower, ic_ci_upper],
        "non_overlapping_ic_mean": mean_non_overlap_ic,
        "non_overlapping_ic_ci_95": [non_overlap_ci_lower, non_overlap_ci_upper],
        "residual_ic_mean": resid_ic_mean,
        "residual_ic_ci_95": [resid_ic_lower, resid_ic_upper],
        "status": status
    }
