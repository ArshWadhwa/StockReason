"""
StockReason v0.8 Data Cleaning, Anomaly Detection & Canonicalization Engine.
==========================================================================
Enforces strict point-in-time invariants:
- Rejects non-positive prices and duplicate timestamps.
- Enforces geometric OHLC consistency: High >= max(Open, Close), Low <= min(Open, Close).
- Detects split/bonus anomalies and preserves raw vs adjusted returns.
- Detects trading suspensions and volume anomalies.
- Produces canonical partitioned datasets adhering to v0.8 contracts.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.v08.contracts import (
    validate_ohlcv_dataframe,
    REQUIRED_OHLCV_COLUMNS,
    CANONICAL_EXTRA_COLUMNS,
)
from src.v08.corporate_actions import compute_consistent_adjusted_ohlcv

logger = logging.getLogger("v08.cleaner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SPLIT_ANOMALY_THRESHOLD = 0.25  # ±25% discontinuity
MAX_RETURN_OUTLIER = 0.60       # 60% single day move without corporate action


def clean_stock_dataframe(
    df: pd.DataFrame,
    ticker: str,
    fix_ohlc_bounds: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a single stock's daily OHLCV dataframe and enriches with canonical fields.
    Returns cleaned DataFrame and cleaning audit statistics dictionary.
    """
    stats = {
        "ticker": ticker,
        "input_rows": len(df),
        "duplicate_rows_removed": 0,
        "non_positive_rows_dropped": 0,
        "ohlc_bounds_adjusted": 0,
        "split_anomalies_detected": 0,
        "suspension_runs_count": 0,
        "output_rows": 0,
    }

    if df.empty:
        empty_res = pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS + CANONICAL_EXTRA_COLUMNS)
        return empty_res, stats

    g = df.copy()

    # 1. Date normalization
    g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    g["ticker"] = str(ticker).strip().upper()

    # 2. Sort and drop duplicates by date
    initial_cnt = len(g)
    g = g.sort_values("date").drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    stats["duplicate_rows_removed"] = initial_cnt - len(g)

    # 3. Filter non-positive prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    valid_prices = (g[price_cols] > 0).all(axis=1)
    bad_price_cnt = (~valid_prices).sum()
    if bad_price_cnt > 0:
        stats["non_positive_rows_dropped"] = int(bad_price_cnt)
        g = g[valid_prices].reset_index(drop=True)

    if g.empty:
        empty_res = pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS + CANONICAL_EXTRA_COLUMNS)
        return empty_res, stats

    # 4. Enforce geometric OHLC consistency
    # High must be >= max(open, close); Low must be <= min(open, close)
    if fix_ohlc_bounds:
        max_oc = np.maximum(g["open"].values, g["close"].values)
        min_oc = np.minimum(g["open"].values, g["close"].values)

        high_adjusted = g["high"].values < max_oc
        low_adjusted = g["low"].values > min_oc
        adjusted_mask = high_adjusted | low_adjusted

        stats["ohlc_bounds_adjusted"] = int(adjusted_mask.sum())
        g["high"] = np.maximum(g["high"].values, max_oc).astype(np.float32)
        g["low"] = np.minimum(g["low"].values, min_oc).astype(np.float32)

    # 5. Non-negative volume
    g["volume"] = np.maximum(g["volume"].fillna(0).values, 0).astype(np.int64)

    # 6. Corporate Action & Consistency enrichment
    g = compute_consistent_adjusted_ohlcv(g)

    # 7. Unadjusted and Adjusted 1-day returns
    close = g["close"]
    adj_close = g["adj_close"]
    prev_close = close.shift(1)
    prev_adj = adj_close.shift(1)

    unadj_ret = ((close - prev_close) / (prev_close + 1e-9)).astype(np.float32)
    adj_ret = ((adj_close - prev_adj) / (prev_adj + 1e-9)).astype(np.float32)

    # Discontinuity check: divergence between unadjusted step and adjusted step
    unadj_ratio = close / (prev_close + 1e-9)
    adj_ratio = adj_close / (prev_adj + 1e-9)
    ratio_diff = (unadj_ratio / (adj_ratio + 1e-9)) - 1.0

    is_split_anomaly = (ratio_diff.abs() >= SPLIT_ANOMALY_THRESHOLD)
    is_split_anomaly.iloc[0] = False  # first observation cannot have step anomaly

    stats["split_anomalies_detected"] = int(is_split_anomaly.sum())

    g["unadj_return_1d"] = unadj_ret.fillna(0.0)
    g["adj_return_1d"] = adj_ret.fillna(0.0)
    g["split_bonus_anomaly"] = is_split_anomaly.fillna(False)

    # 8. Detect suspension runs (>= 5 consecutive zero volume days)
    zero_vol_runs = (g["volume"] == 0).astype(int)
    run_groups = (zero_vol_runs != zero_vol_runs.shift(1)).cumsum()
    run_lengths = zero_vol_runs.groupby(run_groups).transform("sum")
    stats["suspension_runs_count"] = int(((run_lengths >= 5) & (zero_vol_runs == 1)).sum())

    stats["output_rows"] = len(g)
    return g, stats


def clean_universe_panel(
    df_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans an entire multi-ticker OHLCV panel.
    Returns cleaned canonical panel and aggregate quality audit report.
    """
    if df_panel.empty:
        return df_panel, {"total_tickers": 0, "total_rows": 0}

    logger.info(f"Cleaning universe panel with {len(df_panel)} raw rows...")
    cleaned_dfs = []
    overall_stats = {
        "total_tickers": 0,
        "input_rows": len(df_panel),
        "output_rows": 0,
        "duplicate_rows_removed": 0,
        "non_positive_rows_dropped": 0,
        "ohlc_bounds_adjusted": 0,
        "split_anomalies_detected": 0,
        "suspension_runs_count": 0,
        "per_ticker_stats": {},
    }

    for ticker, group in df_panel.groupby("ticker"):
        cleaned_df, stats = clean_stock_dataframe(group, ticker)
        cleaned_dfs.append(cleaned_df)
        overall_stats["total_tickers"] += 1
        overall_stats["duplicate_rows_removed"] += stats["duplicate_rows_removed"]
        overall_stats["non_positive_rows_dropped"] += stats["non_positive_rows_dropped"]
        overall_stats["ohlc_bounds_adjusted"] += stats["ohlc_bounds_adjusted"]
        overall_stats["split_anomalies_detected"] += stats["split_anomalies_detected"]
        overall_stats["suspension_runs_count"] += stats["suspension_runs_count"]
        overall_stats["per_ticker_stats"][ticker] = stats

    result = pd.concat(cleaned_dfs, ignore_index=True) if cleaned_dfs else df_panel
    overall_stats["output_rows"] = len(result)
    logger.info(f"Panel cleaning complete: {overall_stats['output_rows']} canonical rows across {overall_stats['total_tickers']} tickers.")
    return result, overall_stats
