"""
StockReason v0.7 - Evaluation Metrics Library
=============================================
Computes cross-sectional (per-date across stocks) Information Coefficient (IC),
Rank IC (Spearman), ICIR, Rank ICIR, MAE, RMSE, directional accuracy vs base rate,
Brier score, and breakdowns by market volatility regime and industry sector.

Contract requirement (WS10):
compute_metrics(pred_df: pd.DataFrame) -> dict
Keys required:
  ic, rank_ic, icir, rank_icir, mae, rmse, dir_acc, dir_acc_vs_base, brier, n
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_per_date_ic(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    date_col: str = "date",
    min_stocks: int = 2,
) -> pd.DataFrame:
    """
    Computes per-date cross-sectional Pearson IC and Spearman Rank IC across stocks.

    Returns DataFrame with columns: [date, ic, rank_ic, n_stocks].
    """
    records = []
    # Drop rows with NaN in date, pred, or target
    sub = df[[date_col, pred_col, target_col]].dropna()

    for date_val, group in sub.groupby(date_col):
        n_stocks = len(group)
        if n_stocks < min_stocks:
            continue

        p = group[pred_col].to_numpy(dtype=float)
        t = group[target_col].to_numpy(dtype=float)

        std_p = np.std(p)
        std_t = np.std(t)

        if std_p < 1e-12 or std_t < 1e-12:
            # Constant predictions or targets on this date -> 0 correlation
            ic = 0.0
            rank_ic = 0.0
        else:
            r_res = stats.pearsonr(p, t)
            ic = float(r_res.statistic) if hasattr(r_res, "statistic") else float(r_res[0])

            s_res = stats.spearmanr(p, t)
            rank_ic = float(s_res.statistic) if hasattr(s_res, "statistic") else float(s_res[0])

            if np.isnan(ic):
                ic = 0.0
            if np.isnan(rank_ic):
                rank_ic = 0.0

        records.append({
            "date": date_val,
            "ic": ic,
            "rank_ic": rank_ic,
            "n_stocks": n_stocks,
        })

    if not records:
        return pd.DataFrame(columns=["date", "ic", "rank_ic", "n_stocks"])

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric bootstrap confidence interval for the mean of `values`.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps, dtype=float)
    n = len(arr)
    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (lower, upper)


def compute_calibration_metrics(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram bins.
    """
    y_t = np.asarray(y_true_binary, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[mask]
    y_p = np.clip(y_p[mask], 0.0, 1.0)

    n = len(y_t)
    if n == 0:
        return {
            "ece": 0.0,
            "brier": 0.0,
            "bins": [],
        }

    brier = float(np.mean((y_p - y_t) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bin_edges, right=True)

    ece = 0.0
    bins_data = []

    for b in range(1, n_bins + 1):
        idx = (bin_indices == b)
        count = int(np.sum(idx))
        if count > 0:
            avg_prob = float(np.mean(y_p[idx]))
            avg_true = float(np.mean(y_t[idx]))
            weight = count / n
            ece += weight * abs(avg_prob - avg_true)
            bins_data.append({
                "bin": b,
                "bin_lower": float(bin_edges[b - 1]),
                "bin_upper": float(bin_edges[b]),
                "count": count,
                "avg_pred": round(avg_prob, 4),
                "avg_true": round(avg_true, 4),
                "abs_error": round(abs(avg_prob - avg_true), 4),
            })

    return {
        "ece": float(ece),
        "brier": float(brier),
        "bins": bins_data,
    }


def compute_metrics(
    pred_df: pd.DataFrame,
    date_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    target_col: Optional[str] = None,
    prob_col: Optional[str] = None,
    regime_col: Optional[str] = None,
    sector_col: Optional[str] = None,
    min_stocks_per_date: int = 2,
    seed: int = 42,
    include_breakdowns: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive v0.7 quantitative evaluation metrics.

    Parameters:
        pred_df: DataFrame containing predictions and ground truth.
        date_col: Column name for dates. Auto-detected if None.
        pred_col: Column name for predictions. Auto-detected if None.
        target_col: Column name for target values. Auto-detected if None.
        prob_col: Column name for probabilities. Auto-detected if None.
        regime_col: Column name for regime (e.g. TRENDING, VOLATILE, CHOPPY).
        sector_col: Column name for sector.
        min_stocks_per_date: Minimum stocks required per date for cross-sectional IC.
        seed: Random seed for bootstrap confidence intervals.

    Returns:
        dict with keys:
          - ic: float (mean per-date Pearson correlation)
          - rank_ic: float (mean per-date Spearman rank correlation)
          - icir: float (annualized IC Information Ratio: mean(IC)/std(IC)*sqrt(252))
          - rank_icir: float (annualized Rank IC Information Ratio)
          - mae: float (Mean Absolute Error across rows)
          - rmse: float (Root Mean Squared Error across rows)
          - dir_acc: float (Directional Accuracy)
          - dir_acc_vs_base: float (dir_acc - majority class baseline rate)
          - brier: Optional[float] (Brier score if prob available)
          - n: int (number of valid prediction rows)
          - n_dates: int (number of unique evaluated dates)
          - ic_tstat: float (t-statistic for mean IC != 0)
          - rank_ic_tstat: float (t-statistic for mean Rank IC != 0)
          - ic_pval: float (p-value for IC t-stat)
          - rank_ic_pval: float (p-value for Rank IC t-stat)
          - rank_ic_ci_95: Tuple[float, float] (bootstrap 95% CI)
          - by_regime: Dict[str, Dict[str, Any]] (metrics partitioned by regime)
          - by_sector: Dict[str, Dict[str, Any]] (metrics partitioned by sector)
    """
    if pred_df is None or pred_df.empty:
        return {
            "ic": 0.0,
            "rank_ic": 0.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "dir_acc": 0.0,
            "dir_acc_vs_base": 0.0,
            "brier": None,
            "n": 0,
            "n_dates": 0,
            "ic_tstat": 0.0,
            "rank_ic_tstat": 0.0,
            "ic_pval": 1.0,
            "rank_ic_pval": 1.0,
            "rank_ic_ci_95": (0.0, 0.0),
            "by_regime": {},
            "by_sector": {},
        }

    # Auto-resolve columns
    d_col = date_col or _resolve_column(pred_df, ["date", "timestamp", "datetime", "Date", "time"])
    p_col = pred_col or _resolve_column(pred_df, ["y_pred", "pred", "prediction", "signal", "score"])
    t_col = target_col or _resolve_column(pred_df, ["y_true", "actual", "target", "forward_return", "return_1d_fwd", "fwd_return"])
    pr_col = prob_col or _resolve_column(pred_df, ["y_prob", "prob_up_calibrated", "prob_up", "prob", "probability"])
    r_col = regime_col or _resolve_column(pred_df, ["regime", "market_regime", "vol_regime"])
    s_col = sector_col or _resolve_column(pred_df, ["sector", "industry", "sector_name"])

    if not d_col or not p_col or not t_col:
        raise ValueError(
            f"Missing required columns in pred_df. Found: date={d_col}, pred={p_col}, target={t_col}. "
            f"Available columns: {list(pred_df.columns)}"
        )

    # Filter to valid prediction and target rows
    valid_mask = pred_df[p_col].notna() & pred_df[t_col].notna() & pred_df[d_col].notna()
    df_valid = pred_df[valid_mask].copy()

    n_total = len(df_valid)
    if n_total == 0:
        return {
            "ic": 0.0,
            "rank_ic": 0.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "dir_acc": 0.0,
            "dir_acc_vs_base": 0.0,
            "brier": None,
            "n": 0,
            "n_dates": 0,
            "ic_tstat": 0.0,
            "rank_ic_tstat": 0.0,
            "ic_pval": 1.0,
            "rank_ic_pval": 1.0,
            "rank_ic_ci_95": (0.0, 0.0),
            "by_regime": {},
            "by_sector": {},
        }

    p_arr = df_valid[p_col].to_numpy(dtype=float)
    t_arr = df_valid[t_col].to_numpy(dtype=float)

    # 1. Row-level error metrics (MAE, RMSE)
    mae = float(np.mean(np.abs(p_arr - t_arr)))
    rmse = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))

    # 2. Directional Accuracy
    if pr_col and pr_col in df_valid.columns:
        prob_arr = df_valid[pr_col].to_numpy(dtype=float)
        # Handle nan in probabilities if any
        prob_valid = ~np.isnan(prob_arr)
        if np.any(prob_valid):
            pred_dir = (prob_arr[prob_valid] > 0.5).astype(int)
            true_dir = (t_arr[prob_valid] > 0).astype(int)
        else:
            pred_dir = (p_arr > 0).astype(int)
            true_dir = (t_arr > 0).astype(int)
    else:
        pred_dir = (p_arr > 0).astype(int)
        true_dir = (t_arr > 0).astype(int)

    if len(true_dir) > 0:
        dir_acc = float(np.mean(pred_dir == true_dir))
        pos_base = float(np.mean(true_dir == 1))
        class_base_rate = float(max(pos_base, 1.0 - pos_base))
        dir_acc_vs_base = float(dir_acc - class_base_rate)
    else:
        dir_acc = 0.0
        dir_acc_vs_base = 0.0

    # 3. Brier Score
    brier: Optional[float] = None
    if pr_col and pr_col in df_valid.columns:
        prob_arr = df_valid[pr_col].to_numpy(dtype=float)
        mask_pr = ~np.isnan(prob_arr)
        if np.any(mask_pr):
            brier = float(np.mean((prob_arr[mask_pr] - (t_arr[mask_pr] > 0).astype(float)) ** 2))

    # 4. Cross-sectional Per-date IC & Rank IC
    date_metrics_df = compute_per_date_ic(
        df_valid,
        pred_col=p_col,
        target_col=t_col,
        date_col=d_col,
        min_stocks=min_stocks_per_date,
    )

    n_dates = len(date_metrics_df)
    if n_dates == 0:
        ic = 0.0
        rank_ic = 0.0
        icir = 0.0
        rank_icir = 0.0
        ic_tstat = 0.0
        rank_ic_tstat = 0.0
        ic_pval = 1.0
        rank_ic_pval = 1.0
        rank_ic_ci_95 = (0.0, 0.0)
    else:
        ic_series = date_metrics_df["ic"].to_numpy(dtype=float)
        rank_ic_series = date_metrics_df["rank_ic"].to_numpy(dtype=float)

        ic = float(np.mean(ic_series))
        rank_ic = float(np.mean(rank_ic_series))

        ic_std = float(np.std(ic_series, ddof=1)) if n_dates > 1 else 0.0
        rank_ic_std = float(np.std(rank_ic_series, ddof=1)) if n_dates > 1 else 0.0

        ann_factor = np.sqrt(252.0)
        icir = float((ic / (ic_std + 1e-12)) * ann_factor) if ic_std > 1e-12 else 0.0
        rank_icir = float((rank_ic / (rank_ic_std + 1e-12)) * ann_factor) if rank_ic_std > 1e-12 else 0.0

        se_ic = ic_std / np.sqrt(n_dates) if n_dates > 1 else 1e-12
        se_rank_ic = rank_ic_std / np.sqrt(n_dates) if n_dates > 1 else 1e-12

        ic_tstat = float(ic / (se_ic + 1e-12))
        rank_ic_tstat = float(rank_ic / (se_rank_ic + 1e-12))

        if n_dates > 1:
            ic_pval = float(2.0 * (1.0 - stats.t.cdf(abs(ic_tstat), df=n_dates - 1)))
            rank_ic_pval = float(2.0 * (1.0 - stats.t.cdf(abs(rank_ic_tstat), df=n_dates - 1)))
        else:
            ic_pval = 1.0
            rank_ic_pval = 1.0

        rank_ic_ci_95 = compute_bootstrap_ci(rank_ic_series, n_bootstraps=1000, ci=0.95, seed=seed)

    # 5. Breakdown by Regime
    by_regime: Dict[str, Dict[str, Any]] = {}
    if include_breakdowns and r_col and r_col in df_valid.columns:
        for reg_val, reg_group in df_valid.groupby(r_col):
            if len(reg_group) >= min_stocks_per_date:
                # Subcall with breakdowns disabled
                reg_metrics = compute_metrics(
                    reg_group,
                    date_col=d_col,
                    pred_col=p_col,
                    target_col=t_col,
                    prob_col=pr_col,
                    min_stocks_per_date=min_stocks_per_date,
                    seed=seed,
                    include_breakdowns=False,
                )
                by_regime[str(reg_val)] = {
                    "rank_ic": reg_metrics["rank_ic"],
                    "rank_icir": reg_metrics["rank_icir"],
                    "dir_acc": reg_metrics["dir_acc"],
                    "mae": reg_metrics["mae"],
                    "rmse": reg_metrics["rmse"],
                    "n": reg_metrics["n"],
                    "n_dates": reg_metrics["n_dates"],
                }

    # 6. Breakdown by Sector
    by_sector: Dict[str, Dict[str, Any]] = {}
    if include_breakdowns and s_col and s_col in df_valid.columns:
        for sec_val, sec_group in df_valid.groupby(s_col):
            if len(sec_group) >= min_stocks_per_date:
                sec_metrics = compute_metrics(
                    sec_group,
                    date_col=d_col,
                    pred_col=p_col,
                    target_col=t_col,
                    prob_col=pr_col,
                    min_stocks_per_date=min_stocks_per_date,
                    seed=seed,
                    include_breakdowns=False,
                )
                by_sector[str(sec_val)] = {
                    "rank_ic": sec_metrics["rank_ic"],
                    "rank_icir": sec_metrics["rank_icir"],
                    "dir_acc": sec_metrics["dir_acc"],
                    "mae": sec_metrics["mae"],
                    "rmse": sec_metrics["rmse"],
                    "n": sec_metrics["n"],
                    "n_dates": sec_metrics["n_dates"],
                }

    return {
        "ic": round(ic, 5),
        "rank_ic": round(rank_ic, 5),
        "icir": round(icir, 4),
        "rank_icir": round(rank_icir, 4),
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "dir_acc": round(dir_acc, 4),
        "dir_acc_vs_base": round(dir_acc_vs_base, 4),
        "brier": round(brier, 5) if brier is not None else None,
        "n": n_total,
        "n_dates": n_dates,
        "ic_tstat": round(ic_tstat, 3),
        "rank_ic_tstat": round(rank_ic_tstat, 3),
        "ic_pval": round(ic_pval, 5),
        "rank_ic_pval": round(rank_ic_pval, 5),
        "rank_ic_ci_95": (round(rank_ic_ci_95[0], 5), round(rank_ic_ci_95[1], 5)),
        "by_regime": by_regime,
        "by_sector": by_sector,
    }
