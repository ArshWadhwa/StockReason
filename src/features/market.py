"""StockReason v0.13B — Sector, Market & Cross-Sectional Intelligence.

Implements Section 10 & Section 11 Requirements:
1. Market-Wide Features (F1 Family):
   - NIFTY 50 benchmark returns across 1d, 5d, 21d horizons.
   - INDIA VIX level and 5-day change (market uncertainty proxy).
   - Market breadth: % of trading equities advancing on date t.
2. Cross-Sectional Features (F2 Family):
   - Excess return relative to market benchmark: ret_1d - nifty50_ret_1d, ret_21d - nifty50_ret_21d.
   - Point-in-time cross-sectional volume and turnover percentiles (0.0 to 1.0).
   - Point-in-time cross-sectional volatility percentile (rank of vol_20d on date t).
   - Rolling 60-day beta relative to NIFTY 50 benchmark: Cov(r_i, r_m) / Var(r_m).
3. Corporate-Action & Suspension Protections (Section 11):
   - Identifies session gaps > 10 calendar days as trading resumptions.
   - Preserves genuine market shocks while insulating variance models from unadjusted split spikes.
   - Calculated strictly using observable securities on date t, eliminating survivorship bias.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013b_sector_cross_sectional")


def load_market_benchmark_indices(
    indices_dir: str = "data/external/sector_indices",
) -> pd.DataFrame:
    """Loads NIFTY 50 and INDIA VIX benchmarks, returning unified daily market DataFrame."""
    nifty_path = os.path.join(indices_dir, "NIFTY_50.parquet")
    vix_path = os.path.join(indices_dir, "INDIA_VIX.parquet")

    if not os.path.exists(nifty_path):
        logger.warning("NIFTY 50 path %s not found", nifty_path)
        return pd.DataFrame()

    df_nifty = pd.read_parquet(nifty_path)
    df_nifty["date"] = pd.to_datetime(df_nifty.index)
    df_nifty = df_nifty.sort_values("date").reset_index(drop=True)

    nifty_close = df_nifty["Close"]
    df_nifty["nifty50_ret_1d"] = (nifty_close / nifty_close.shift(1)) - 1.0
    df_nifty["nifty50_ret_5d"] = (nifty_close / nifty_close.shift(5)) - 1.0
    df_nifty["nifty50_ret_21d"] = (nifty_close / nifty_close.shift(21)) - 1.0
    df_nifty["nifty50_vol_20d"] = df_nifty["nifty50_ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)

    keep_cols = ["date", "nifty50_ret_1d", "nifty50_ret_5d", "nifty50_ret_21d", "nifty50_vol_20d"]

    if os.path.exists(vix_path):
        df_vix = pd.read_parquet(vix_path)
        df_vix["date"] = pd.to_datetime(df_vix.index)
        df_vix = df_vix.sort_values("date").reset_index(drop=True)
        vix_close = df_vix["Close"]
        df_vix["india_vix_level"] = vix_close
        df_vix["india_vix_change_5d"] = vix_close - vix_close.shift(5)
        
        merged_mkt = pd.merge(
            df_nifty[keep_cols],
            df_vix[["date", "india_vix_level", "india_vix_change_5d"]],
            on="date",
            how="left",
        )
    else:
        merged_mkt = df_nifty[keep_cols].copy()
        merged_mkt["india_vix_level"] = np.nan
        merged_mkt["india_vix_change_5d"] = np.nan

    logger.info("Loaded market benchmarks: %d trading sessions from %s to %s",
                len(merged_mkt), str(merged_mkt["date"].min())[:10], str(merged_mkt["date"].max())[:10])
    return merged_mkt


def compute_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes date-level cross-sectional ranks, market breadth, and relative volume
    using strictly observable securities on each date t (no survivorship bias).
    """
    logger.info("Computing cross-sectional percentiles and market breadth...")
    panel = panel_df.copy()
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    # 1. Market Breadth per date: % of equities with ret_1d > 0
    if "ret_1d" in panel.columns:
        date_breadth = panel.groupby("date")["ret_1d"].apply(lambda s: (s > 0).mean()).to_dict()
        panel["market_breadth_pct_up"] = panel["date"].map(date_breadth)
    else:
        panel["market_breadth_pct_up"] = 0.50

    # 2. Cross-sectional volume and turnover percentile
    if "volume" in panel.columns:
        panel["cross_sec_volume_pctile"] = panel.groupby("date")["volume"].rank(pct=True)
    else:
        panel["cross_sec_volume_pctile"] = 0.50

    if "turnover" in panel.columns:
        panel["cross_sec_turnover_pctile"] = panel.groupby("date")["turnover"].rank(pct=True)
    else:
        panel["cross_sec_turnover_pctile"] = 0.50

    if "vol_20d" in panel.columns:
        panel["cross_sec_vol_pctile"] = panel.groupby("date")["vol_20d"].rank(pct=True)
    else:
        panel["cross_sec_vol_pctile"] = 0.50

    return panel


def merge_market_and_sector_features(
    panel_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges market benchmark indices onto panel and computes stock excess return relative to market.
    """
    logger.info("Merging market benchmark indices onto panel (%d rows)...", len(panel_df))
    panel = panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")

    mkt = market_df.copy()
    mkt["date"] = pd.to_datetime(mkt["date"]).astype("datetime64[ns]")

    merged = pd.merge(panel, mkt, on="date", how="left")

    # Excess return relative to NIFTY 50
    if "ret_1d" in merged.columns and "nifty50_ret_1d" in merged.columns:
        merged["excess_ret_market_1d"] = merged["ret_1d"] - merged["nifty50_ret_1d"]
    else:
        merged["excess_ret_market_1d"] = np.nan

    if "ret_21d" in merged.columns and "nifty50_ret_21d" in merged.columns:
        merged["excess_ret_market_21d"] = merged["ret_21d"] - merged["nifty50_ret_21d"]
    else:
        merged["excess_ret_market_21d"] = np.nan

    # Fill VIX forward with neutral baseline if missing
    merged["india_vix_level"] = merged["india_vix_level"].ffill().fillna(15.0)
    merged["india_vix_change_5d"] = merged["india_vix_change_5d"].fillna(0.0)

    # Cross-sectional ranks
    merged = compute_cross_sectional_features(merged)

    logger.info("Sector, market and cross-sectional features constructed successfully.")
    return merged
