"""
StockReason v0.10 — Advanced Cross-Sectional & Market Structure Feature Engine.
================================================================================
Remediates Phase 7 (Section 13):
1. Daily Cross-Sectional Percentiles (0.0 to 1.0):
   - Momentum Percentile: Cross-sectional rank of trailing 21d return.
   - Volatility Percentile: Cross-sectional rank of trailing 60d volatility.
   - Liquidity Percentile: Cross-sectional rank of trailing turnover (volume * close).
2. Market Structure Features:
   - Market Breadth: Fraction of active universe stocks trading above 50-day moving average.
   - Cross-Sectional Return Dispersion: Daily cross-sectional standard deviation of returns.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_daily_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-sectional percentile ranks and market structure features per date.
    """
    out = panel_df.copy()
    if "date_str" not in out.columns:
        out["date_str"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    # 1. Trailing turnover proxy
    if "turnover" in out.columns and not out["turnover"].isna().all():
        out["turnover_val"] = out["turnover"].fillna(0.0)
    else:
        out["turnover_val"] = out["close"] * out["volume"]

    # 2. Daily cross-sectional percentiles using groupby transform
    out["mom_pctile_21d"] = out.groupby("date_str")["ret_21d"].rank(pct=True).fillna(0.5) if "ret_21d" in out.columns else 0.5
    out["vol_pctile_60d"] = out.groupby("date_str")["vol_60d"].rank(pct=True).fillna(0.5) if "vol_60d" in out.columns else 0.5
    out["turnover_pctile"] = out.groupby("date_str")["turnover_val"].rank(pct=True).fillna(0.5)

    # 3. Market breadth: stocks above close_sma_50
    if "close_sma_50" in out.columns:
        out["is_above_sma50"] = (out["close"] > out["close_sma_50"]).astype(float)
        breadth_series = out.groupby("date_str")["is_above_sma50"].transform("mean")
        out["market_breadth_sma50"] = breadth_series.fillna(0.5)
    else:
        out["market_breadth_sma50"] = 0.5

    # 4. Cross-sectional dispersion of 21d returns
    if "ret_21d" in out.columns:
        disp_series = out.groupby("date_str")["ret_21d"].transform("std")
        out["cs_dispersion_21d"] = disp_series.fillna(0.02)
    else:
        out["cs_dispersion_21d"] = 0.02

    return out
