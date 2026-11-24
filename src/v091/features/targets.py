"""
StockReason v0.9.1 — Cross-Sectional Demeaned Target Construction.
==================================================================
Remediates the core model collapse defect of v0.9:
- v0.9 used raw forward return (target_return_21d).
  Because training had mean +1.94% and validation had mean +3.61%, MSE early stopping
  halted at iteration 2 fitting an unconditional positive intercept drift.
- v0.9.1 constructs:
  1. `target_demeaned_21d`: Cross-sectional demeaned forward return per date:
     y_demeaned_{i, t} = y_{i, t} - mean_{j}(y_{j, t})
     Strictly removes market drift and forces cross-sectional dispersion.
  2. `target_vol_scaled_21d`: Demeaned return normalized by trailing volatility:
     y_vol_{i, t} = y_demeaned_{i, t} / (vol_60d_{i, t} + 1e-4)
  3. `target_rank_21d`: Cross-sectional uniform percentile rank in [0, 1].
- Decomposes price forecast into:
  Price_{t+H} = Price_t * (1.0 + r_market_component + r_stock_residual)
  where r_market_component is explicitly estimated and reported separately.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def construct_cross_sectional_targets(
    df: pd.DataFrame,
    horizons: List[int] = [1, 5, 21, 63, 126],
) -> pd.DataFrame:
    """
    Computes cross-sectional demeaned, volatility-scaled, and ranked forward targets
    strictly within each observation date.
    """
    df_out = df.copy()
    if "date_str" not in df_out.columns:
        df_out["date_str"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
        
    for h in horizons:
        raw_col = f"target_return_{h}d"
        if raw_col not in df_out.columns:
            continue
            
        demeaned_col = f"target_demeaned_{h}d"
        ranked_col = f"target_rank_{h}d"
        vol_scaled_col = f"target_vol_scaled_{h}d"
        
        # 1. Cross-sectional mean per date
        date_means = df_out.groupby("date_str")[raw_col].transform("mean")
        df_out[demeaned_col] = df_out[raw_col] - date_means
        
        # 2. Cross-sectional percentile rank per date
        # (Uniformly distributed in [0, 1])
        df_out[ranked_col] = df_out.groupby("date_str")[raw_col].rank(pct=True)
        
        # 3. Volatility scaled demeaned return
        vol_col = "vol_60d" if "vol_60d" in df_out.columns else ("vol_20d" if "vol_20d" in df_out.columns else None)
        if vol_col:
            df_out[vol_scaled_col] = df_out[demeaned_col] / (df_out[vol_col] + 1e-4)
            
    return df_out


def decompose_price_forecast(
    ref_price: float,
    pred_residual: float,
    market_return_forecast: float = 0.015, # trailing Nifty 21d baseline
) -> Dict[str, float]:
    """
    Decomposes price forecast into market drift and stock-specific residual:
    Price_{t+21} = Price_t * (1 + r_market + r_residual)
    """
    total_expected_return = market_return_forecast + pred_residual
    expected_price = ref_price * (1.0 + total_expected_return)
    return {
        "reference_price_inr": ref_price,
        "market_component_return": market_return_forecast,
        "residual_component_return": pred_residual,
        "total_expected_return": total_expected_return,
        "expected_price_inr": expected_price
    }
