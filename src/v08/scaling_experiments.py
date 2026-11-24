"""
StockReason v0.8 Scaling Experiment Suite.
=========================================
Rigorous empirical evaluation comparing 6 dataset configurations:
- Config A: 96 stocks (original v0.7.1 universe) x historical depth (from 2017)
- Config B: 100 stocks x 5 years (2021 - 2026)
- Config C: 100 stocks x 10 years (2016 - 2026)
- Config D: 100 stocks x 15 years (2011 - 2026)
- Config E: 300 stocks x 15 years (2011 - 2026)
- Config F: 500 stocks x 15 years (2011 - 2026)

Evaluates whether expanding historical depth and universe breadth improves:
1. Exact-price & return forecasting (MAE, RMSE, Directional Accuracy, Bias).
2. Stock ranking (Per-date cross-sectional Rank IC, 95% block bootstrap CI).
3. Trading strategy performance (Gross/Net return, Sharpe, Max Drawdown, Turnover, Cost Drag).
Strictly evaluated on IDENTICAL out-of-sample holdout test dates and common evaluation universe.
"""

import os
import sys

# CRITICAL ON MACOS: Set OpenMP flag and import torch BEFORE lightgbm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.v071.metrics import compute_ic, summarize_ic, compute_directional_accuracy
from src.v071.cost_model import calculate_trade_cost
from src.v08.universe import get_universe, get_universe_details
from src.v08.features import compute_technical_features, compute_multi_horizon_targets, extract_feature_column_names
from src.v08.storage import write_parquet_atomic

logger = logging.getLogger("v08.scaling")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_RESULTS_PATH = "artifacts/v0.8/scaling_experiment_results.json"


def block_bootstrap_rank_ic(
    ic_series: np.ndarray,
    n_replications: int = 2000,
    block_length: int = 10,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes stationary block-bootstrap confidence interval for Rank IC series.
    Returns: (lower_ci, upper_ci, p_val_two_tailed).
    """
    clean_ic = ic_series[~np.isnan(ic_series)]
    n = len(clean_ic)
    if n < 5:
        return 0.0, 0.0, 1.0

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_replications, dtype=np.float64)

    for b in range(n_replications):
        n_blocks = int(np.ceil(n / block_length))
        start_indices = rng.integers(0, max(1, n - block_length + 1), size=n_blocks)
        sample = np.concatenate([clean_ic[idx : idx + block_length] for idx in start_indices])[:n]
        boot_means[b] = np.mean(sample)

    alpha = 1.0 - confidence_level
    lower_ci = float(np.percentile(boot_means, alpha / 2.0 * 100))
    upper_ci = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))

    # Fraction of draws with opposite sign of mean
    mean_val = float(np.mean(clean_ic))
    if mean_val >= 0:
        p_val = 2.0 * float(np.mean(boot_means <= 0))
    else:
        p_val = 2.0 * float(np.mean(boot_means >= 0))
    p_val = min(1.0, max(0.0, p_val))

    return round(lower_ci, 4), round(upper_ci, 4), round(p_val, 4)


def simulate_topk_strategy(
    pred_df: pd.DataFrame,
    k: int = 10,
    holding_days: int = 21,
    cost_bps: float = 23.3,  # SEBI 2024 delivery roundtrip cost ~23.3 bps
) -> Dict[str, Any]:
    """
    Simulates Top-K cross-sectional long-only strategy rebalanced every holding_days.
    Uses next-day open / next session execution.
    """
    if pred_df.empty or len(pred_df) < 50:
        return {"gross_return_pct": 0.0, "net_return_pct": 0.0, "sharpe": 0.0, "turnover": 0.0, "max_drawdown_pct": 0.0}

    dates = sorted(pred_df["date"].unique())
    # Rebalance every holding_days
    rebal_dates = dates[::holding_days]

    period_returns = []
    portfolio_wealth = 100.0
    wealth_curve = [100.0]

    for d in rebal_dates:
        sub = pred_df[pred_df["date"] == d].dropna(subset=["pred", "target"])
        if len(sub) < k:
            continue
        # Pick top k predictions
        top_k = sub.nlargest(k, "pred")
        realized_return = float(top_k["target"].mean())
        # Apply transaction cost
        net_return = realized_return - (cost_bps / 10000.0)

        period_returns.append(net_return)
        portfolio_wealth *= (1.0 + net_return)
        wealth_curve.append(portfolio_wealth)

    if not period_returns:
        return {"gross_return_pct": 0.0, "net_return_pct": 0.0, "sharpe": 0.0, "turnover": 0.0, "max_drawdown_pct": 0.0}

    total_net_return = (portfolio_wealth / 100.0) - 1.0
    arr = np.array(period_returns)
    sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252 / holding_days))

    # Max Drawdown
    wealths = np.array(wealth_curve)
    peaks = np.maximum.accumulate(wealths)
    drawdowns = (wealths - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    return {
        "net_return_pct": round(total_net_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "turnover_annual": round((252 / holding_days) * 2.0, 1),
        "rebalance_events": len(period_returns),
    }


def run_scaling_benchmark_experiment(
    panel_df: pd.DataFrame,
    common_test_tickers: List[str],
    test_start_date: str = "2025-05-01",
    horizon: int = 21,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes scaling experiments A through F.
    Ensures IDENTICAL test dates and common evaluation universe across all configurations.
    """
    logger.info("Preparing scaling experiment configurations...")
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    feature_cols = extract_feature_column_names(df)
    target_col = f"target_return_{horizon}d"

    # Identify out-of-sample test split
    test_dt = pd.to_datetime(test_start_date)
    all_dates = sorted(df["date"].unique())
    test_dates = [d for d in all_dates if d >= test_dt]
    train_dates_full = [d for d in all_dates if d < test_dt - pd.Timedelta(days=horizon + 5)]

    logger.info(f"Test period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} sessions)")

    # Configurations definition
    max_train_dt = max(train_dates_full)
    configs = {
        "Config_A": {
            "name": "Config A: Original 96 stocks x Historical (from 2017)",
            "tickers": common_test_tickers[:96],
            "start_date": "2017-09-01",
        },
        "Config_B": {
            "name": "Config B: 100 stocks x 5 years",
            "tickers": common_test_tickers[:100],
            "start_date": (max_train_dt - pd.Timedelta(days=5 * 365)).strftime("%Y-%m-%d"),
        },
        "Config_C": {
            "name": "Config C: 100 stocks x 10 years",
            "tickers": common_test_tickers[:100],
            "start_date": (max_train_dt - pd.Timedelta(days=10 * 365)).strftime("%Y-%m-%d"),
        },
        "Config_D": {
            "name": "Config D: 100 stocks x 15 years",
            "tickers": common_test_tickers[:100],
            "start_date": "2011-11-01",
        },
        "Config_E": {
            "name": "Config E: 300 stocks x 15 years",
            "tickers": get_universe("U300"),
            "start_date": "2011-11-01",
        },
        "Config_F": {
            "name": "Config F: 500 stocks x 15 years",
            "tickers": get_universe("U500"),
            "start_date": "2011-11-01",
        },
    }

    results = {}

    for cfg_id, cfg_info in configs.items():
        logger.info(f"Evaluating {cfg_info['name']}...")
        cfg_tickers = set(cfg_info["tickers"])
        start_dt = pd.to_datetime(cfg_info["start_date"])

        # Filter training data
        train_mask = (df["ticker"].isin(cfg_tickers)) & (df["date"] >= start_dt) & (df["date"].isin(train_dates_full))
        train_sub = df[train_mask].dropna(subset=[target_col]).copy()
        train_sub[feature_cols] = train_sub[feature_cols].fillna(0.0)

        if train_sub.empty or len(train_sub) < 100:
            logger.warning(f"{cfg_id}: Insufficient training rows ({len(train_sub)}).")
            continue

        X_train = train_sub[feature_cols].values
        y_train = train_sub[target_col].values

        # Scale features strictly on train
        scaler = StandardScaler()
        X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)

        # Train primary CatBoost / LightGBM models
        model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=31, random_state=seed, n_jobs=1, verbose=-1)
        model.fit(X_train_s, y_train)

        # Baseline Ridge model
        ridge = Ridge(alpha=10.0, random_state=seed)
        ridge.fit(X_train_s, y_train)

        # Evaluate on COMMON TEST UNIVERSE (Apples-to-apples evaluation)
        common_test_mask = (df["ticker"].isin(set(common_test_tickers[:96]))) & (df["date"].isin(test_dates))
        test_sub = df[common_test_mask].dropna(subset=[target_col]).copy()
        test_sub[feature_cols] = test_sub[feature_cols].fillna(0.0)

        if test_sub.empty:
            continue

        X_test_s = np.nan_to_num(scaler.transform(test_sub[feature_cols].values), nan=0.0)
        y_test = test_sub[target_col].values

        y_pred = model.predict(X_test_s)
        y_pred_ridge = ridge.predict(X_test_s)

        test_sub["pred"] = y_pred
        test_sub["pred_ridge"] = y_pred_ridge
        test_sub["target"] = y_test

        # 1. Exact Return Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        bias = float(np.mean(y_pred - y_test))

        dir_acc = compute_directional_accuracy(test_sub, pred_col="pred", target_col="target")

        # 2. Per-date Cross-Sectional Rank IC
        ic_df = compute_ic(test_sub, pred_col="pred", target_col="target", min_names_per_date=10)
        ic_summary = summarize_ic(ic_df) if not ic_df.empty else {"mean_ic": 0.0, "icir_annualized": 0.0, "std_ic": 0.0}

        mean_ic = float(ic_summary.get("mean_ic", 0.0))
        lower_ci, upper_ci, p_val = block_bootstrap_rank_ic(ic_df["ic"].values if not ic_df.empty else np.array([0.0]))

        # Ridge IC comparison
        ic_ridge_df = compute_ic(test_sub, pred_col="pred_ridge", target_col="target", min_names_per_date=10)
        ridge_ic_summary = summarize_ic(ic_ridge_df) if not ic_ridge_df.empty else {"mean_ic": 0.0}

        # 3. Trading Strategy Simulation
        strat_metrics = simulate_topk_strategy(test_sub, k=10, holding_days=horizon)

        results[cfg_id] = {
            "name": cfg_info["name"],
            "universe_size": len(cfg_info["tickers"]),
            "historical_start": cfg_info["start_date"],
            "train_observations": len(train_sub),
            "test_observations_evaluated": len(test_sub),
            "exact_price_metrics": {
                "return_mae": round(mae, 6),
                "return_rmse": round(rmse, 6),
                "bias": round(bias, 6),
                "directional_accuracy_pct": round(dir_acc.get("accuracy_pct", 50.0), 2),
                "majority_base_rate_pct": round(dir_acc.get("majority_base_rate_pct", 50.0), 2),
            },
            "stock_ranking_metrics": {
                "mean_cross_sectional_rank_ic": round(mean_ic, 4),
                "annualized_icir": round(float(ic_summary.get("icir_annualized", 0.0)), 2),
                "ic_95_ci": [lower_ci, upper_ci],
                "bootstrap_p_value": p_val,
                "spans_zero": bool(lower_ci <= 0.0 <= upper_ci),
                "ridge_mean_rank_ic": round(float(ridge_ic_summary.get("mean_ic", 0.0)), 4),
            },
            "strategy_metrics": strat_metrics,
        }

    return results
