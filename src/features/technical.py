"""F0: Price and Volume Technical Indicators Module."""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_technical_features(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
) -> pd.DataFrame:
    """
    Computes verified F0 Core Price and Volume technical indicators (17 features):
    - ret_1d, ret_5d, ret_10d, ret_21d, ret_63d, ret_126d
    - vol_20d, vol_60d (annualized realized volatility)
    - dist_ma20, dist_ma50, dist_ma200 (distance from moving averages)
    - rsi_14 (Relative Strength Index)
    - macd_diff (MACD difference: EMA12 - EMA26 - Signal9)
    - hl_spread ((High - Low) / Close)
    - vol_ratio_20d (Volume / 20d Volume MA)
    - log_turnover (log(Turnover in INR + 1))
    """
    out = df.copy()
    if not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col])

    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)
    grp = out.groupby(symbol_col)

    # 1. Multi-horizon historical returns
    out["ret_1d"] = grp[close_col].pct_change(1)
    out["ret_5d"] = grp[close_col].pct_change(5)
    out["ret_10d"] = grp[close_col].pct_change(10)
    out["ret_21d"] = grp[close_col].pct_change(21)
    out["ret_63d"] = grp[close_col].pct_change(63)
    out["ret_126d"] = grp[close_col].pct_change(126)

    # 2. Historical annualized realized volatility
    out["vol_20d"] = grp["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(252)
    out["vol_60d"] = grp["ret_1d"].transform(lambda s: s.rolling(60, min_periods=30).std()) * np.sqrt(252)

    # 3. Moving average distances
    ma20 = grp[close_col].transform(lambda s: s.rolling(20, min_periods=10).mean())
    ma50 = grp[close_col].transform(lambda s: s.rolling(50, min_periods=25).mean())
    ma200 = grp[close_col].transform(lambda s: s.rolling(200, min_periods=100).mean())

    out["dist_ma20"] = (out[close_col] / ma20) - 1.0
    out["dist_ma50"] = (out[close_col] / ma50) - 1.0
    out["dist_ma200"] = (out[close_col] / ma200) - 1.0

    # 4. RSI (14-day)
    def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100.0 - (100.0 / (1.0 + rs))

    out["rsi_14"] = grp[close_col].transform(calc_rsi)

    # 5. MACD (12, 26, 9)
    def calc_macd_diff(series: pd.Series) -> pd.Series:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    out["macd_diff"] = grp[close_col].transform(calc_macd_diff)

    # 6. High-Low Spread
    if high_col in out.columns and low_col in out.columns:
        out["hl_spread"] = (out[high_col] - out[low_col]) / out[close_col]
    else:
        out["hl_spread"] = 0.0

    # 7. Volume ratio
    if volume_col in out.columns:
        vol_ma20 = grp[volume_col].transform(lambda s: s.rolling(20, min_periods=10).mean())
        out["vol_ratio_20d"] = out[volume_col] / (vol_ma20 + 1e-6)
    else:
        out["vol_ratio_20d"] = 1.0

    # 8. Log turnover
    if turnover_col in out.columns:
        out["log_turnover"] = np.log1p(np.maximum(out[turnover_col].fillna(0.0), 0.0))
    elif volume_col in out.columns:
        out["log_turnover"] = np.log1p(np.maximum(out[close_col] * out[volume_col], 0.0))
    else:
        out["log_turnover"] = 0.0

    return out
