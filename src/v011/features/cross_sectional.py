"""
StockReason v0.11 — Cross-Sectional & Relative Feature Intelligence.
====================================================================
Remediates Phase 3 (Section 19):
1. Daily point-in-time percentile ranking across all active securities on date t.
2. Cross-Sectional Features:
   - cs_mom_rank_21d: Short-term momentum percentile [0.0, 1.0]
   - cs_mom_rank_126d: Intermediate momentum percentile [0.0, 1.0]
   - cs_vol_rank_60d: Volatility percentile [0.0, 1.0]
   - cs_turnover_rank_20d: Liquidity / turnover percentile [0.0, 1.0]
   - cs_rel_strength_rank: Market-relative outperformance percentile
3. Prevents lookahead: Ranking is performed strictly within each date group.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_daily_cross_sectional_features(df: pd.DataFrame, date_col: str = "date_str") -> pd.DataFrame:
    """
    Computes daily cross-sectional percentile rankings for active equities on date t.
    """
    out = df.copy()
    if date_col not in out.columns:
        out[date_col] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    # 1. 21D Momentum Rank
    if "ret_21d" in out.columns:
        out["cs_mom_rank_21d"] = out.groupby(date_col)["ret_21d"].rank(pct=True).fillna(0.5)
    else:
        out["cs_mom_rank_21d"] = 0.5

    # 2. 126D Momentum Rank
    if "ret_126d" in out.columns:
        out["cs_mom_rank_126d"] = out.groupby(date_col)["ret_126d"].rank(pct=True).fillna(0.5)
    else:
        out["cs_mom_rank_126d"] = 0.5

    # 3. Volatility Rank
    if "vol_60d" in out.columns:
        out["cs_vol_rank_60d"] = out.groupby(date_col)["vol_60d"].rank(pct=True).fillna(0.5)
    else:
        out["cs_vol_rank_60d"] = 0.5

    # 4. Turnover Rank
    if "log_turnover" in out.columns:
        out["cs_turnover_rank_20d"] = out.groupby(date_col)["log_turnover"].rank(pct=True).fillna(0.5)
    else:
        out["cs_turnover_rank_20d"] = 0.5

    # 5. Relative Strength Rank
    if "rel_strength_21d" in out.columns:
        out["cs_rel_strength_rank"] = out.groupby(date_col)["rel_strength_21d"].rank(pct=True).fillna(0.5)
    else:
        out["cs_rel_strength_rank"] = 0.5

    return out
