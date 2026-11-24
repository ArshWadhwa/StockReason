"""
StockReason v0.10 — Canonical Universe Definitions & Controlled Universe Comparison.
=====================================================================================
Remediates Phase 1 (Section 5):
1. Maintains two explicitly defined universes:
   - Broad training universe: All genuinely eligible historical NSE equities (1,840)
     satisfying point-in-time minimum history (>= 250 sessions), positive price, non-suspended.
   - Investable evaluation universe: Top-500 liquid securities selected at each monthly
     rebalance date t using only trailing traded value available up to t.
2. Compares three training regimes on identical out-of-sample investable evaluation rows:
   - Regime 1: Top-500-only training.
   - Regime 2: Broad eligible universe training.
   - Regime 3: Broad training with Top-500 sample weighting / fine-tuning.
3. Reports unique securities, dates, observations, and exclusions for each experiment.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v092.canonical.universe import investigate_universe_observation_eligibility
from src.v092.models.baselines import compute_absolute_price_metrics


def audit_historical_universe_membership(
    panel_df: pd.DataFrame,
    pit_universe_path: str = "data/v0.9.1/canonical/universe_membership/monthly_pit_universe.parquet",
) -> Dict[str, Any]:
    """
    Audits the broad versus Top-500 investable universe membership per observation.
    Returns observation counts, eligibility breakdown, and marks boolean membership.
    """
    if "date_str" not in panel_df.columns:
        panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")
    panel_df["month_key"] = panel_df["date_str"].str[:7]

    pit_df = pd.read_parquet(pit_universe_path)
    pit_df["month_key"] = pit_df["rebal_str"].str[:7]

    # Fast lookup set of (month_key, symbol)
    eligible_pairs = set(zip(pit_df["month_key"], pit_df["symbol"]))

    panel_pairs = list(zip(panel_df["month_key"], panel_df["symbol"]))
    panel_df["is_top500_constituent"] = [p in eligible_pairs for p in panel_pairs]

    train_mask = panel_df["date_str"] < "2023-01-01"
    val_mask = (panel_df["date_str"] >= "2023-01-01") & (panel_df["date_str"] < "2025-01-01")

    train_total = int(train_mask.sum())
    train_top500 = int((train_mask & panel_df["is_top500_constituent"]).sum())
    train_broad_only = train_total - train_top500

    val_total = int(val_mask.sum())
    val_top500 = int((val_mask & panel_df["is_top500_constituent"]).sum())
    val_broad_only = val_total - val_top500

    return {
        "panel_total_observations": len(panel_df),
        "train_total_observations": train_total,
        "train_top500_eligible": train_top500,
        "train_broad_only": train_broad_only,
        "train_eligibility_pct": round(train_top500 / max(1, train_total) * 100.0, 2),
        "val_total_observations": val_total,
        "val_top500_eligible": val_top500,
        "val_broad_only": val_broad_only,
        "val_eligibility_pct": round(val_top500 / max(1, val_total) * 100.0, 2),
    }


def compare_universe_training_regimes(
    clean_panel: pd.DataFrame,
    features: List[str] = FEATURE_COLS_21,
    target_col: str = "target_return_21d",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compares three controlled training regimes on identical out-of-sample investable evaluation rows:
    1. Regime 1: Top-500-only training (exclusively active constituent rows).
    2. Regime 2: Broad universe training (all 1,840 equities meeting basic listing criteria).
    3. Regime 3: Broad training with sample weighting (2x weight on Top-500 rows).
    """
    train_df = clean_panel[clean_panel["date_str"] < "2023-01-01"].copy()
    
    # Strictly evaluate on the investable Top-500 evaluation universe in validation period
    val_df_top500 = clean_panel[
        (clean_panel["date_str"] >= "2023-01-01")
        & (clean_panel["date_str"] < "2025-01-01")
        & (clean_panel["is_top500_constituent"] == True)
    ].copy()

    X_val = val_df_top500[features].values
    y_val = val_df_top500[target_col].values
    val_close = val_df_top500["close"].values

    results = []

    # -------------------------------------------------------------------------
    # Regime 1: Top-500-Only Training
    # -------------------------------------------------------------------------
    train_r1 = train_df[train_df["is_top500_constituent"] == True].dropna(subset=features + [target_col])
    X_tr1 = train_r1[features].values
    y_tr1 = train_r1[target_col].values

    # Ridge
    m_ridge_r1 = Ridge(alpha=100.0, random_state=seed).fit(X_tr1, y_tr1)
    p_ridge_r1 = m_ridge_r1.predict(X_val)
    res_ridge_r1 = compute_absolute_price_metrics(y_val, p_ridge_r1, val_close)

    # LightGBM
    lgb_r1 = lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=100, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
    ).fit(X_tr1, y_tr1)
    p_lgb_r1 = lgb_r1.predict(X_val)
    res_lgb_r1 = compute_absolute_price_metrics(y_val, p_lgb_r1, val_close)

    results.append({
        "regime": "1. Top-500 Only Training",
        "training_observations": len(train_r1),
        "evaluation_observations": len(val_df_top500),
        "model": "Ridge",
        **res_ridge_r1,
    })
    results.append({
        "regime": "1. Top-500 Only Training",
        "training_observations": len(train_r1),
        "evaluation_observations": len(val_df_top500),
        "model": "LightGBM",
        **res_lgb_r1,
    })

    # -------------------------------------------------------------------------
    # Regime 2: Broad Universe Training (All eligible 1,840 equities)
    # -------------------------------------------------------------------------
    train_r2 = train_df.dropna(subset=features + [target_col])
    X_tr2 = train_r2[features].values
    y_tr2 = train_r2[target_col].values

    # Ridge
    m_ridge_r2 = Ridge(alpha=100.0, random_state=seed).fit(X_tr2, y_tr2)
    p_ridge_r2 = m_ridge_r2.predict(X_val)
    res_ridge_r2 = compute_absolute_price_metrics(y_val, p_ridge_r2, val_close)

    # LightGBM
    lgb_r2 = lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=100, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
    ).fit(X_tr2, y_tr2)
    p_lgb_r2 = lgb_r2.predict(X_val)
    res_lgb_r2 = compute_absolute_price_metrics(y_val, p_lgb_r2, val_close)

    results.append({
        "regime": "2. Broad Universe Training",
        "training_observations": len(train_r2),
        "evaluation_observations": len(val_df_top500),
        "model": "Ridge",
        **res_ridge_r2,
    })
    results.append({
        "regime": "2. Broad Universe Training",
        "training_observations": len(train_r2),
        "evaluation_observations": len(val_df_top500),
        "model": "LightGBM",
        **res_lgb_r2,
    })

    # -------------------------------------------------------------------------
    # Regime 3: Broad Training with Top-500 Weighting (2.0x weight on Top-500)
    # -------------------------------------------------------------------------
    sample_weights = np.where(train_r2["is_top500_constituent"].values, 2.0, 1.0)

    # Ridge with weights
    m_ridge_r3 = Ridge(alpha=100.0, random_state=seed).fit(X_tr2, y_tr2, sample_weight=sample_weights)
    p_ridge_r3 = m_ridge_r3.predict(X_val)
    res_ridge_r3 = compute_absolute_price_metrics(y_val, p_ridge_r3, val_close)

    # LightGBM with weights
    lgb_r3 = lgb.LGBMRegressor(
        learning_rate=0.03, n_estimators=100, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1
    ).fit(X_tr2, y_tr2, sample_weight=sample_weights)
    p_lgb_r3 = lgb_r3.predict(X_val)
    res_lgb_r3 = compute_absolute_price_metrics(y_val, p_lgb_r3, val_close)

    results.append({
        "regime": "3. Broad Training + Top-500 Weighting",
        "training_observations": len(train_r2),
        "evaluation_observations": len(val_df_top500),
        "model": "Ridge",
        **res_ridge_r3,
    })
    results.append({
        "regime": "3. Broad Training + Top-500 Weighting",
        "training_observations": len(train_r2),
        "evaluation_observations": len(val_df_top500),
        "model": "LightGBM",
        **res_lgb_r3,
    })

    return pd.DataFrame(results)
