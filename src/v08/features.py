"""
StockReason v0.8 Feature Engineering and Multi-Horizon Target Builder.
=====================================================================
Features:
- Strictly backward-looking technical indicators computed per-security (data <= t).
- Multi-horizon trailing returns (1d, 5d, 21d, 63d).
- Volatilities, ATR, Moving Average distances, Momentum, RSI, MACD, Bollinger Bands.

Targets (for horizons h in [1, 5, 21, 63, 126]):
- target_return_{h}d: (adj_close[t+h] / adj_close[t]) - 1.0
- target_price_{h}d: close[t+h] (tradable future share price)
- target_direction_{h}d: (target_return_{h}d > 0).astype(int)
- target_rank_{h}d: cross-sectional percentile rank normalized to [-0.5, +0.5] per date.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("v08.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SUPPORTED_HORIZONS = [1, 5, 21, 63, 126]


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly backward-looking technical features per ticker.
    Ensures zero forward lookahead leakage.
    """
    if df.empty:
        return df

    out_groups = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        close = g["close"]
        adj_close = g["adj_close"]
        high = g["high"]
        low = g["low"]
        vol = g["volume"].fillna(0)

        # 1. Trailing Returns (using adj_close to account for splits/dividends)
        for p in [1, 2, 3, 5, 10, 15, 21, 42, 63]:
            g[f"return_{p}d"] = ((adj_close - adj_close.shift(p)) / (adj_close.shift(p) + 1e-9)).astype(np.float32)
            g[f"log_return_{p}d"] = np.log(adj_close / (adj_close.shift(p) + 1e-9)).astype(np.float32)

        # 2. Realized Volatility (annualized, sqrt(252))
        ret_1d = g["return_1d"]
        g["vol_20d"] = (ret_1d.rolling(20).std() * np.sqrt(252)).astype(np.float32)
        g["vol_60d"] = (ret_1d.rolling(60).std() * np.sqrt(252)).astype(np.float32)

        # 3. ATR and NATR (14-day)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["atr_14"] = tr.rolling(14).mean().astype(np.float32)
        g["natr_14"] = ((g["atr_14"] / (close + 1e-9)) * 100.0).astype(np.float32)

        # 4. Moving Average Distances
        for w in [20, 50, 200]:
            sma = close.rolling(w).mean()
            g[f"dist_sma_{w}"] = ((close - sma) / (sma + 1e-9)).astype(np.float32)

        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        g["sma_50_200_ratio"] = (sma_50 / (sma_200 + 1e-9)).astype(np.float32)

        # 5. RSI (14 and 28-day)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        for rsi_period in [14, 28]:
            avg_gain = gain.rolling(rsi_period).mean()
            avg_loss = loss.rolling(rsi_period).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            g[f"rsi_{rsi_period}"] = (100.0 - (100.0 / (1.0 + rs))).astype(np.float32)

        # 6. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        g["macd"] = macd_line.astype(np.float32)
        g["macd_signal"] = signal_line.astype(np.float32)
        g["macd_hist"] = (macd_line - signal_line).astype(np.float32)

        # 7. Bollinger Bands (20-day, 2 std)
        rolling_mean_20 = close.rolling(20).mean()
        rolling_std_20 = close.rolling(20).std()
        upper_bb = rolling_mean_20 + 2.0 * rolling_std_20
        lower_bb = rolling_mean_20 - 2.0 * rolling_std_20
        g["bb_width_20"] = ((upper_bb - lower_bb) / (rolling_mean_20 + 1e-9)).astype(np.float32)
        g["bb_pct_b_20"] = ((close - lower_bb) / (upper_bb - lower_bb + 1e-9)).astype(np.float32)

        # 8. Volume Dynamics
        vol_sma_20 = vol.rolling(20).mean()
        g["vol_ratio_20"] = (vol / (vol_sma_20 + 1e-9)).astype(np.float32)

        out_groups.append(g)

    result = pd.concat(out_groups, ignore_index=True)
    return result


def compute_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: List[int] = SUPPORTED_HORIZONS,
) -> pd.DataFrame:
    """
    Computes future return, future price, and direction targets for horizons h in `horizons`.
    Formula:
    target_return_{h}d = (adj_close[t+h] - adj_close[t]) / adj_close[t]
    target_price_{h}d = close[t+h]
    target_direction_{h}d = (target_return_{h}d > 0).astype(int)
    """
    out_groups = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        adj_close = g["adj_close"]
        close = g["close"]

        for h in horizons:
            # Shift backwards to align future observation with current decision date t
            future_adj = adj_close.shift(-h)
            future_close = close.shift(-h)

            ret = (future_adj - adj_close) / (adj_close + 1e-9)
            g[f"target_return_{h}d"] = ret.astype(np.float32)
            g[f"target_price_{h}d"] = future_close.astype(np.float32)
            g[f"target_direction_{h}d"] = (ret > 0).astype(np.float32)

        out_groups.append(g)

    result = pd.concat(out_groups, ignore_index=True)

    # Compute cross-sectional normalized rank per date for each horizon
    for h in horizons:
        ret_col = f"target_return_{h}d"
        rank_col = f"target_rank_{h}d"
        # rank within each trading date
        ranks = result.groupby("date")[ret_col].rank(pct=True, ascending=True) - 0.5
        result[rank_col] = ranks.astype(np.float32)

    return result


def extract_feature_column_names(df: pd.DataFrame) -> List[str]:
    """Returns list of feature column names (excluding raw prices, targets, ticker, date)."""
    excluded_prefixes = ["target_", "adj_open", "adj_high", "adj_low", "adj_volume"]
    excluded_exact = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "sector", "split_bonus_anomaly"]

    cols = []
    for c in df.columns:
        if c in excluded_exact:
            continue
        if any(c.startswith(p) for p in excluded_prefixes):
            continue
        cols.append(c)
    return sorted(cols)
