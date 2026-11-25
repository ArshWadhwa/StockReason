"""Liquidity, Delivery & Trading Dynamics Module."""

import numpy as np
import pandas as pd


def compute_liquidity_features(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    delivery_pct_col: str = "delivery_pct",
) -> pd.DataFrame:
    """
    Computes liquidity and delivery dynamics:
    - abnormal_delivery_ratio: delivery_pct / 20d moving average
    - delivery_pct_spread: delivery_pct - 20d moving average
    - turnover_ratio_20d: turnover / 20d moving average
    - amihud_illiquidity: |ret_1d| / (turnover / 1e7)
    """
    out = df.copy()
    if not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col])

    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)
    grp = out.groupby(symbol_col)

    if delivery_pct_col in out.columns:
        deliv_ma20 = grp[delivery_pct_col].transform(lambda s: s.rolling(20, min_periods=5).mean())
        out["abnormal_delivery_ratio"] = out[delivery_pct_col] / (deliv_ma20 + 1e-6)
        out["delivery_pct_spread"] = out[delivery_pct_col] - deliv_ma20

    if turnover_col in out.columns:
        turn_ma20 = grp[turnover_col].transform(lambda s: s.rolling(20, min_periods=5).mean())
        out["turnover_20d_ma"] = turn_ma20
        out["turnover_ratio_20d"] = out[turnover_col] / (turn_ma20 + 1e-6)

        if "ret_1d" in out.columns:
            # Amihud illiquidity in units of 10M INR
            out["amihud_illiquidity"] = np.abs(out["ret_1d"]) / ((out[turnover_col] / 1e7) + 1e-4)

    return out
