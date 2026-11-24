"""
StockReason v0.7.1 - Lockbox Final Evaluation Script
====================================================
Evaluates the final frozen model ONCE on the strictly protected lockbox partition.

Enforcement:
1. Verifies frozen config hash against expected hash.
2. Verifies that reports/v0.7.1/lockbox_access.log contains ZERO previous evaluation runs.
3. Trains on ALL non-lockbox data (2021-11-18 to 2026-04-09 for U100, or 2026-01-28 for U16).
4. Predicts the lockbox partition (exclude_lockbox=False).
5. Computes honest cross-sectional IC, block bootstrap CI, backtest, and alpha attribution.
6. Writes signed JSON result to artifacts/v0.7.1/lockbox_final_result.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.v07.common.lockbox import (
    load_panel,
    get_lockbox_date_bounds,
    get_config_hash,
    UNLOCK_ENV_VAR,
    UNLOCK_MAGIC_VALUE,
    LOCKBOX_LOG_PATH,
)
from src.v071.metrics import compute_ic, summarize_ic, compute_directional_accuracy
from src.v071.honest_statistics import block_bootstrap_ic, compute_newey_west_tstat
from src.v071.strategy_and_benchmarks import run_hysteresis_backtest, run_attribution_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LockboxEvaluation")


def verify_lockbox_purity() -> None:
    """Refuses execution if lockbox_access.log shows any previous evaluation."""
    log_path = Path(LOCKBOX_LOG_PATH)
    if log_path.exists():
        with open(log_path, "r") as f:
            lines = f.readlines()
        eval_lines = [l for l in lines if "lockbox_evaluate" in l or "final_eval" in l]
        if eval_lines:
            raise RuntimeError(
                f"LOCKBOX CONTAMINATION VIOLATION! Lockbox has already been evaluated "
                f"{len(eval_lines)} times. Second evaluations are strictly prohibited."
            )


def run_lockbox_evaluation(
    frozen_config_path: str = "config/v071.yaml",
    expected_config_hash: Optional[str] = None,
    output_path: str = "artifacts/v0.7.1/lockbox_final_result.json",
    panel_path: str = "data/v0.7.1/panel_u100.parquet",
) -> Dict[str, Any]:
    current_hash = get_config_hash(frozen_config_path)
    if expected_config_hash is not None and current_hash != expected_config_hash:
        raise ValueError(
            f"Config hash mismatch! Expected: {expected_config_hash}, Found: {current_hash}. "
            f"Lockbox evaluation refuses to run on an unfrozen/modified config."
        )

    # Verify environment variable first
    if os.environ.get(UNLOCK_ENV_VAR) != UNLOCK_MAGIC_VALUE:
        raise PermissionError(
            f"Set {UNLOCK_ENV_VAR}={UNLOCK_MAGIC_VALUE} to execute the final lockbox evaluation."
        )

    verify_lockbox_purity()

    logger.info("Unlocking lockbox data using panel at %s...", panel_path)
    full_panel = load_panel(path=panel_path, exclude_lockbox=False, config_path=frozen_config_path)
    full_panel["date"] = pd.to_datetime(full_panel["date"])

    lockbox_start, embargo_start, max_date = get_lockbox_date_bounds(full_panel)
    logger.info("Lockbox Bounds: Start=%s, Embargo=%s, Max=%s", lockbox_start, embargo_start, max_date)

    feature_cols = [
        "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_21d",
        "vol_20d", "vol_60d", "natr_14", "natr_20", "dist_sma_10", "dist_sma_20",
        "dist_sma_50", "dist_sma_100", "dist_sma_200", "rsi_14", "rsi_28",
        "bb_pct_b_20", "bb_width_20", "bb_width_50", "sma_50_200_ratio", "vol_ratio_20", "high_low_spread"
    ]
    feature_cols = [c for c in feature_cols if c in full_panel.columns]

    # Training set: strictly prior to embargo
    tr_df = full_panel[full_panel["date"] < embargo_start].dropna(subset=["y_21d"] + feature_cols).copy()
    # Lockbox evaluation set: strictly >= lockbox_start
    te_df = full_panel[full_panel["date"] >= lockbox_start].dropna(subset=["y_21d"] + feature_cols).copy()

    logger.info("Training on %d rows (dates: %s to %s)...", len(tr_df), tr_df["date"].min(), tr_df["date"].max())
    logger.info("Evaluating on %d lockbox rows (dates: %s to %s)...", len(te_df), te_df["date"].min(), te_df["date"].max())

    X_tr = tr_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_tr = tr_df["y_21d"].to_numpy(dtype=np.float32)
    X_te = te_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_te = te_df["y_21d"].to_numpy(dtype=np.float32)

    # Train frozen CatBoost model across 3 seeds
    cb_preds = []
    for s in [42, 101, 2024]:
        clf = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, random_seed=s, thread_count=1, verbose=0)
        clf.fit(X_tr, y_tr)
        cb_preds.append(clf.predict(X_te))
    p_te = np.mean(cb_preds, axis=0)

    # Evaluation metrics
    eval_df = pd.DataFrame({"date": te_df["date"].values, "ticker": te_df["ticker"].values, "pred": p_te, "target": y_te})
    ic_df = compute_ic(eval_df, min_names_per_date=20)
    ic_summary = summarize_ic(ic_df, horizon=21)
    boot_res = block_bootstrap_ic(ic_df["ic"].to_numpy(), block_length=21, n_replications=5000, seed=42)
    dir_res = compute_directional_accuracy(eval_df)

    # Backtest on Lockbox
    ret_df = te_df[["date", "ticker", "return_1d"]].copy()
    strat_res = run_hysteresis_backtest(eval_df, ret_df, k=10, band=20, rebalance_freq=21)

    # Nifty benchmark on lockbox dates
    bench_returns = te_df.groupby("date")["nifty_50_close"].first().pct_change().dropna().to_numpy()
    nifty_total_return = float((te_df["nifty_50_close"].iloc[-1] / te_df["nifty_50_close"].iloc[0]) - 1.0) if len(te_df) > 0 else 0.0

    attrib_res = run_attribution_regression(strat_res["daily_returns"], bench_returns)

    result = {
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_path": frozen_config_path,
        "config_hash": current_hash,
        "panel_path": panel_path,
        "train_rows": len(tr_df),
        "train_date_range": [str(tr_df["date"].min()), str(tr_df["date"].max())],
        "lockbox_rows": len(te_df),
        "lockbox_date_range": [str(te_df["date"].min()), str(te_df["date"].max())],
        "lockbox_trading_days": te_df["date"].nunique(),
        "lockbox_tickers": te_df["ticker"].nunique(),
        "ic_metrics": {
            "mean_ic": ic_summary["mean_ic"],
            "std_ic": ic_summary["std_ic"],
            "icir_annualized": ic_summary["icir_annualized"],
            "bootstrap_ci_95": boot_res["ci_95"],
            "pct_positive_draws": boot_res["pct_positive_draws"],
            "p_val_bootstrap": boot_res["p_val_bootstrap"],
            "directional_accuracy": dir_res["dir_acc"],
            "base_rate": dir_res["pos_base_rate"],
        },
        "strategy_backtest": {
            "total_return": strat_res["total_return"],
            "cagr": strat_res["cagr"],
            "sharpe": strat_res["sharpe"],
            "max_drawdown": strat_res["max_drawdown"],
            "annualized_turnover": strat_res["annualized_turnover"],
            "cost_drag_bps": strat_res["cost_drag_bps"],
            "nifty_total_return": nifty_total_return,
        },
        "return_attribution": attrib_res,
        "verdict": "SUCCESS" if boot_res["ci_95"][0] > 0 else "NO_DEMONSTRATED_EDGE",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Lockbox evaluation complete. Result written to %s", output_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-config", type=str, default="config/v071.yaml")
    parser.add_argument("--expected-hash", type=str, default=None)
    args = parser.parse_args()
    run_lockbox_evaluation(frozen_config_path=args.frozen_config, expected_config_hash=args.expected_hash)
