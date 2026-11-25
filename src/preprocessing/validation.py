"""Feature Quality, Availability & Collinearity Validation Module."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def audit_feature_availability(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Audits feature completeness, non-null counts, unique values, and distributions.
    """
    rows = []
    total_len = len(df)

    for col in feature_cols:
        if col not in df.columns:
            rows.append({
                "feature": col,
                "present": False,
                "coverage_pct": 0.0,
                "missing_count": total_len,
                "unique_values": 0,
                "mean": np.nan,
                "std": np.nan,
            })
            continue

        s = df[col]
        non_null = int(s.notna().sum())
        cov = (non_null / total_len * 100.0) if total_len > 0 else 0.0
        uniq = int(s.nunique(dropna=True))

        valid_vals = s.dropna()
        if pd.api.types.is_numeric_dtype(s) and len(valid_vals) > 0:
            mean_val = float(valid_vals.mean())
            std_val = float(valid_vals.std())
        else:
            mean_val = np.nan
            std_val = np.nan

        rows.append({
            "feature": col,
            "present": True,
            "coverage_pct": round(cov, 2),
            "missing_count": total_len - non_null,
            "unique_values": uniq,
            "mean": round(mean_val, 4) if pd.notna(mean_val) else np.nan,
            "std": round(std_val, 4) if pd.notna(std_val) else np.nan,
        })

    return pd.DataFrame(rows)


def check_constant_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    """Returns list of features that are completely constant (nunique <= 1)."""
    constant_cols = []
    for col in feature_cols:
        if col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)
    return constant_cols


def check_feature_collinearity(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Identifies pairs of features with absolute Pearson correlation exceeding threshold.
    """
    num_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return []

    # Sample if large dataset to preserve memory
    sample_df = df[num_cols].sample(min(100000, len(df)), random_state=42) if len(df) > 100000 else df[num_cols]
    corr_mat = sample_df.corr().abs()

    collinear_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1 = num_cols[i]
            c2 = num_cols[j]
            val = corr_mat.loc[c1, c2]
            if val >= threshold:
                collinear_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(val), 4),
                })

    return collinear_pairs
