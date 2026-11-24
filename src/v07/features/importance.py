"""
StockReason v0.7 Feature Redundancy & Importance Analysis Module.

Provides:
1. Pairwise feature correlation matrix (Pearson & Spearman).
2. Redundancy clustering (|corr| > 0.9) to detect and prune collinear feature duplicates.
3. Multi-fold LightGBM Gain Importance with Purged Time-Series cross-validation.
4. Out-of-fold Permutation Feature Importance evaluated on Rank IC.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix across numeric features.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    sub_df = df[valid_cols].dropna(how="all")
    return sub_df.corr(method=method)


def cluster_redundant_features(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.90,
    target_corrs: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Identify clusters of features with absolute correlation >= threshold.
    Recommends keeping the single highest-value feature per cluster and dropping redundant ones.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix of features.
    threshold : float, default 0.90
        Absolute correlation threshold for redundancy.
    target_corrs : Optional[pd.Series]
        Feature correlations with target to guide representative selection.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - clusters: List of feature lists grouped by correlation
          - redundant_pairs: List of (f1, f2, corr)
          - recommended_keep: List of features to keep
          - recommended_drop: List of redundant features to drop
    """
    cols = list(corr_matrix.columns)
    abs_corr = corr_matrix.abs()

    # Find connected components of |corr| >= threshold
    adj: Dict[str, Set[str]] = {c: set() for c in cols}
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = abs_corr.loc[c1, c2]
            if r >= threshold and not np.isnan(r):
                adj[c1].add(c2)
                adj[c2].add(c1)
                pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "abs_corr": float(r),
                    "raw_corr": float(corr_matrix.loc[c1, c2]),
                })

    visited: Set[str] = set()
    clusters: List[List[str]] = []
    recommended_drop: List[str] = []
    recommended_keep: List[str] = []

    for c in cols:
        if c not in visited:
            # BFS to find cluster
            cluster = []
            queue = [c]
            visited.add(c)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(cluster) > 1:
                clusters.append(sorted(cluster))
                # Select best representative: highest abs target correlation if available, else first
                if target_corrs is not None:
                    scored = [(f, abs(target_corrs.get(f, 0.0))) for f in cluster]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    best_f = scored[0][0]
                else:
                    best_f = cluster[0]

                recommended_keep.append(best_f)
                for f in cluster:
                    if f != best_f:
                        recommended_drop.append(f)
            else:
                recommended_keep.append(c)

    return {
        "threshold": threshold,
        "total_features": len(cols),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "redundant_pair_count": len(pairs),
        "redundant_pairs": sorted(pairs, key=lambda x: x["abs_corr"], reverse=True),
        "recommended_keep": sorted(recommended_keep),
        "recommended_drop": sorted(recommended_drop),
    }


def compute_lgbm_and_permutation_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "y_5d",
    date_col: str = "date",
    n_splits: int = 3,
    embargo_days: int = 5,
    random_state: int = 42,
    lgb_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train LightGBM across purged chronological time-series folds to compute:
      - LightGBM Gain Importance (mean & std across folds)
      - Permutation Feature Importance (drop in validation Spearman Rank IC)

    Strictly prevents lookahead leakage: models are trained only on past folds
    with an embargo window between train and test.
    """
    valid_features = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_features:
        raise ValueError("No valid numeric feature columns provided for importance analysis.")

    clean_df = df.dropna(subset=[target_col] + valid_features).copy()
    if clean_df.empty:
        raise ValueError(f"No valid rows remaining after dropping NaNs in target '{target_col}' and features.")

    # Unique dates for chronological splitting
    unique_dates = np.sort(clean_df[date_col].unique())
    if len(unique_dates) < n_splits + 1:
        raise ValueError(f"Insufficient unique dates ({len(unique_dates)}) for {n_splits} splits.")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if lgb_params is None:
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": random_state,
            "n_estimators": 150,
            "n_jobs": -1,
        }

    fold_gains: List[Dict[str, float]] = []
    fold_splits: List[Dict[str, float]] = []
    fold_perm_ic_drops: List[Dict[str, float]] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(unique_dates)):
        train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        # Apply embargo: drop last embargo_days from train_dates to prevent target overlap
        if embargo_days > 0 and len(train_dates) > embargo_days:
            train_dates = train_dates[:-embargo_days]

        train_mask = clean_df[date_col].isin(train_dates)
        val_mask = clean_df[date_col].isin(val_dates)

        x_train = clean_df.loc[train_mask, valid_features]
        y_train = clean_df.loc[train_mask, target_col]
        x_val = clean_df.loc[val_mask, valid_features]
        y_val = clean_df.loc[val_mask, target_col]

        if x_train.empty or x_val.empty:
            continue

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(x_train, y_train)

        # Baseline evaluation
        preds_val = model.predict(x_val)
        baseline_ic, _ = spearmanr(preds_val, y_val)
        if np.isnan(baseline_ic):
            baseline_ic = 0.0
        val_mse = float(np.mean((preds_val - y_val) ** 2))

        fold_metrics.append({
            "fold": fold + 1,
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "rank_ic": float(baseline_ic),
            "mse": float(val_mse),
        })

        # Feature Gain Importance
        booster = model.booster_
        gain_imp = booster.feature_importance(importance_type="gain")
        split_imp = booster.feature_importance(importance_type="split")

        fold_gains.append({f: float(g) for f, g in zip(valid_features, gain_imp)})
        fold_splits.append({f: float(s) for f, s in zip(valid_features, split_imp)})

        # Permutation Importance on validation set (evaluating delta IC)
        perm_drops: Dict[str, float] = {}
        # Shuffle top 30 features by gain to keep runtime fast and focused
        top_gain_feats = [f for f, _ in sorted(zip(valid_features, gain_imp), key=lambda x: x[1], reverse=True)[:30]]
        rng = np.random.RandomState(random_state + fold)
        
        for feat in top_gain_feats:
            x_val_perm = x_val.copy()
            x_val_perm[feat] = rng.permutation(x_val_perm[feat].values)
            perm_preds = model.predict(x_val_perm)
            perm_ic, _ = spearmanr(perm_preds, y_val)
            if np.isnan(perm_ic):
                perm_ic = 0.0
            ic_drop = float(baseline_ic - perm_ic)
            perm_drops[feat] = ic_drop

        fold_perm_ic_drops.append(perm_drops)

    # Aggregate across folds
    df_gain = pd.DataFrame(fold_gains)
    df_split = pd.DataFrame(fold_splits)
    df_perm = pd.DataFrame(fold_perm_ic_drops).fillna(0.0)

    summary_rows = []
    for f in valid_features:
        gain_mean = float(df_gain[f].mean()) if f in df_gain.columns else 0.0
        gain_std = float(df_gain[f].std()) if f in df_gain.columns else 0.0
        split_mean = float(df_split[f].mean()) if f in df_split.columns else 0.0
        perm_mean = float(df_perm[f].mean()) if f in df_perm.columns else 0.0
        perm_std = float(df_perm[f].std()) if f in df_perm.columns else 0.0

        summary_rows.append({
            "feature": f,
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "split_mean": split_mean,
            "perm_ic_drop_mean": perm_mean,
            "perm_ic_drop_std": perm_std,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("gain_mean", ascending=False).reset_index(drop=True)

    return {
        "target": target_col,
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "summary": summary_df.to_dict(orient="records"),
        "top_10_gain": summary_df.head(10)[["feature", "gain_mean", "perm_ic_drop_mean"]].to_dict(orient="records"),
    }
