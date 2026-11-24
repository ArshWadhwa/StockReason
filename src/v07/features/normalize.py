"""
StockReason v0.7 Cross-Sectional Normalization Module.

Provides robust cross-sectional normalization across equity tickers on each trading date.

Supported Methods:
1. "rank":
   Rank-transforms each feature cross-sectionally per date into [-0.5, 0.5]:
     norm_rank = (rank - 1) / (N_t - 1) - 0.5
   Guarantees:
     - Uniform cross-sectional distribution
     - Strict bounds [-0.5, 0.5]
     - Exact mean = 0.0 across assets on each date
     - Complete immunity to extreme outlier leverage

2. "zscore":
   Standardizes cross-sectionally per date:
     z = (x - mean_t) / (std_t + eps)
   Optionally clipped to [clip_min, clip_max] (e.g. [-3.0, 3.0]).
   Guarantees:
     - Zero mean and unit variance cross-sectionally

3. "none":
   Pass-through without transformation.
"""

from typing import List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_transform_group(
    vals: pd.Series,
    out_bounds: Tuple[float, float] = (-0.5, 0.5),
) -> pd.Series:
    """
    Transform a single date cross-sectional group into out_bounds via rank normalization.
    """
    valid = vals.notna()
    n = valid.sum()
    if n == 0:
        return vals.copy()
    if n == 1:
        res = pd.Series(np.nan, index=vals.index)
        res[valid] = 0.0
        return res

    min_b, max_b = out_bounds
    span = max_b - min_b
    ranks = vals[valid].rank(method="average", ascending=True)
    if ranks.max() == ranks.min():
        # All values equal in cross-section
        norm_ranks = pd.Series(0.0, index=ranks.index)
    else:
        norm_ranks = ((ranks - 1.0) / (n - 1.0)) * span + min_b

    res = pd.Series(np.nan, index=vals.index)
    res[valid] = norm_ranks
    return res


def zscore_transform_group(
    vals: pd.Series,
    eps: float = 1e-8,
    clip_bounds: Optional[Tuple[float, float]] = (-3.0, 3.0),
) -> pd.Series:
    """
    Transform a single date cross-sectional group via z-score standardization.
    """
    valid = vals.notna()
    n = valid.sum()
    if n <= 1:
        res = pd.Series(np.nan, index=vals.index)
        if n == 1:
            res[valid] = 0.0
        return res

    subset = vals[valid]
    mu = subset.mean()
    sigma = subset.std(ddof=1)
    if sigma < eps or np.isnan(sigma):
        z = pd.Series(0.0, index=subset.index)
    else:
        z = (subset - mu) / sigma

    if clip_bounds is not None:
        z = z.clip(clip_bounds[0], clip_bounds[1])

    res = pd.Series(np.nan, index=vals.index)
    res[valid] = z
    return res


def normalize_cross_section(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "date",
    method: str = "rank",
    out_bounds: Tuple[float, float] = (-0.5, 0.5),
    clip_zscore: Optional[Tuple[float, float]] = (-3.0, 3.0),
) -> pd.DataFrame:
    """
    Normalize features cross-sectionally per date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing date_col and feature_cols.
    feature_cols : List[str]
        List of numeric feature column names to normalize.
    date_col : str, default 'date'
        Column name for date.
    method : str, default 'rank'
        Normalization method: 'rank', 'zscore', or 'none'.
    out_bounds : Tuple[float, float], default (-0.5, 0.5)
        Bounds for rank normalization.
    clip_zscore : Optional[Tuple[float, float]], default (-3.0, 3.0)
        Outlier clip bounds for zscore normalization.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with normalized feature columns.
    """
    norm_method = method.strip().lower()
    if norm_method == "none":
        return df.copy()
    if norm_method not in {"rank", "zscore"}:
        raise ValueError(f"Unsupported normalization method '{method}'. Expected 'rank', 'zscore', or 'none'.")

    if df.empty:
        return df.copy()

    result = df.copy()
    valid_features = [c for c in feature_cols if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_features:
        logger.warning("No valid numeric feature columns provided for cross-sectional normalization.")
        return result

    logger.info(f"Applying cross-sectional '{norm_method}' normalization to {len(valid_features)} features across dates...")

    if norm_method == "rank":
        min_b, max_b = out_bounds
        span = max_b - min_b

        def _rank_block(block: pd.DataFrame) -> pd.DataFrame:
            out = block.copy()
            for col in valid_features:
                s = block[col]
                valid = s.notna()
                n = valid.sum()
                if n <= 1:
                    out.loc[valid, col] = 0.0
                else:
                    ranks = s[valid].rank(method="average", ascending=True)
                    if ranks.max() == ranks.min():
                        out.loc[valid, col] = 0.0
                    else:
                        out.loc[valid, col] = ((ranks - 1.0) / (n - 1.0)) * span + min_b
            return out

        # Groupby transform on chunks
        result[valid_features] = (
            result.groupby(date_col, group_keys=False)[valid_features]
            .apply(_rank_block)
        )

    elif norm_method == "zscore":
        def _zscore_block(block: pd.DataFrame) -> pd.DataFrame:
            out = block.copy()
            for col in valid_features:
                s = block[col]
                valid = s.notna()
                n = valid.sum()
                if n <= 1:
                    out.loc[valid, col] = 0.0
                else:
                    sub = s[valid]
                    mu = sub.mean()
                    sig = sub.std(ddof=1)
                    if sig < 1e-8 or np.isnan(sig):
                        out.loc[valid, col] = 0.0
                    else:
                        z = (sub - mu) / sig
                        if clip_zscore is not None:
                            z = z.clip(clip_zscore[0], clip_zscore[1])
                        out.loc[valid, col] = z
            return out

        result[valid_features] = (
            result.groupby(date_col, group_keys=False)[valid_features]
            .apply(_zscore_block)
        )

    return result
