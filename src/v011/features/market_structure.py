"""
StockReason v0.11 — Market Structure & Delivery Intelligence.
==============================================================
Remediates Phase 3 (Section 18):
1. Ingests delivery data (NSE delivery volume, delivery percentage).
2. Market microstructure & price behavior features:
   - ret_overnight: (Open_t / Close_{t-1}) - 1.0
   - ret_intraday: (Close_t / Open_t) - 1.0
   - hl_spread: (High_t - Low_t) / Close_t
   - turnover_shock: log(turnover) - mean_20d(log(turnover))
   - delivery_pct: Realized delivery volume / total trading volume
   - delivery_trend_5d: delivery_pct / rolling_20d(delivery_pct)
   - downside_semivariance: semi-variance of negative daily returns over 20 days
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_market_structure_features(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """
    Computes microstructure and delivery-related indicators on daily OHLCV panel.
    """
    out = df.sort_values([symbol_col, "date"]).copy()

    # 1. Overnight and Intraday returns
    prev_close = out["prev_close"] if "prev_close" in out.columns else out.groupby(symbol_col)["close"].shift(1)
    prev_close = np.where(prev_close <= 0, out["open"], prev_close)

    out["ret_overnight"] = np.clip((out["open"] / prev_close) - 1.0, -0.30, 0.30)
    out["ret_intraday"] = np.clip((out["close"] / np.maximum(out["open"], 1e-4)) - 1.0, -0.30, 0.30)

    # 2. High-Low Spread
    out["hl_spread"] = np.clip((out["high"] - out["low"]) / np.maximum(out["close"], 1e-4), 0.0, 0.50)

    # 3. Turnover Shock
    if "log_turnover" not in out.columns and "turnover" in out.columns:
        out["log_turnover"] = np.log1p(np.maximum(out["turnover"], 0.0))

    if "log_turnover" in out.columns:
        mean_to_20 = out.groupby(symbol_col)["log_turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        std_to_20 = out.groupby(symbol_col)["log_turnover"].transform(lambda s: s.rolling(20, min_periods=5).std())
        out["turnover_shock_20d"] = np.clip((out["log_turnover"] - mean_to_20) / np.maximum(std_to_20, 1e-3), -3.0, 3.0)
    else:
        out["turnover_shock_20d"] = 0.0

    # 4. Delivery Percentage (if present or proxy from volume stability)
    if "delivery_pct" in out.columns:
        mean_del = out.groupby(symbol_col)["delivery_pct"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        out["delivery_trend_5d"] = np.clip(out["delivery_pct"] / np.maximum(mean_del, 1.0), 0.2, 5.0)
    else:
        out["delivery_pct"] = 40.0 # Default median delivery % on NSE
        out["delivery_trend_5d"] = 1.0

    # 5. Downside Semivariance over 20 days
    ret_1d = out["ret_1d"] if "ret_1d" in out.columns else out.groupby(symbol_col)["close"].pct_change()
    out["_neg_ret_sq"] = np.where(ret_1d < 0, ret_1d ** 2, 0.0)
    out["downside_semivar_20d"] = out.groupby(symbol_col)["_neg_ret_sq"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    ) * 252.0 # Annualized
    out = out.drop(columns=["_neg_ret_sq"])

    return out
