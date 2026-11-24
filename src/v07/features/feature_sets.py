"""
StockReason v0.7 Feature Sets & Stationarity Enforcement Module.

Defines the 5 ablation feature sets (A, B, C, D, E) and provides strict stationarity
validation to guarantee zero non-stationary price/volume levels leak into ML models.

Ablation Sets:
- Set A: Minimal stationary technicals (returns, relative momentum, realized vol, ATR%, volume ratios)
- Set B: Set A + Market and sector context (Nifty return, VIX level/change, macro, RS vs Nifty & sector)
- Set C: Set B + Stationary oscillators (RSI, normalized MACD hist, Bollinger %B, ADX, distance-to-MA)
- Set D: Set C + News & NLP features (FinBERT sentiment mean, momentum, divergence)
- Set E: Control set (All 216 legacy features, including non-stationary levels)

Stationarity Rules for Sets A-D:
Strictly EXCLUDES:
  - Raw price levels: open, high, low, close, adj_close
  - Raw moving averages: sma_*, ema_* (nominal levels)
  - Raw nominal ATR: atr_*
  - Raw volume & cumulative volume: volume, vol_sma_*, volume_lag_*, obv, pvt
  - Raw MACD price lines: macd_line_*, macd_signal_*
  - Raw channel levels: bb_upper_*, bb_lower_*, donchian_high_*, keltner_upper_*
Strictly USES:
  - Percentage distances: dist_sma_*, dist_ema_*
  - Normalized ATR: natr_* (ATR / Close * 100)
  - Normalized volume: vol_ratio_* (Volume / Volume SMA)
  - Normalized MACD histogram: macd_hist_*_norm (MACD Hist / Close)
  - Bounded oscillators: rsi_*, stoch_*, bb_pct_b_*, adx_*, donchian_pos_*
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Strict prohibited columns for stationary sets (A, B, C, D)
PROHIBITED_STATIONARY_COLUMNS: Set[str] = {
    # Raw price levels
    "open", "high", "low", "close", "adj_close",
    # Raw volume & cumulative volume
    "volume", "obv", "obv_ema_20", "pvt",
    # Raw macro/index price levels
    "nifty_50_close", "nifty_bank_close", "nifty_it_close",
}

# Regex patterns matching non-stationary nominal level columns
PROHIBITED_PATTERNS: List[re.Pattern] = [
    re.compile(r"^sma_\d+$"),                     # e.g., sma_5, sma_20, sma_200
    re.compile(r"^ema_\d+$"),                     # e.g., ema_5, ema_20, ema_200
    re.compile(r"^atr_\d+$"),                     # e.g., atr_7, atr_14
    re.compile(r"^atr_\d+_lag_\d+$"),             # e.g., atr_14_lag_1
    re.compile(r"^vol_sma_\d+$"),                 # e.g., vol_sma_20
    re.compile(r"^volume_lag_\d+$"),              # e.g., volume_lag_1
    re.compile(r"^macd_line_\d+_\d+$"),           # e.g., macd_line_12_26
    re.compile(r"^macd_signal_\d+_\d+$"),         # e.g., macd_signal_12_26
    re.compile(r"^macd_line_\d+_\d+_lag_\d+$"),   # e.g., macd_line_12_26_lag_1
    re.compile(r"^bb_upper_\d+$"),                # e.g., bb_upper_20
    re.compile(r"^bb_lower_\d+$"),                # e.g., bb_lower_20
    re.compile(r"^bb_mid_\d+$"),                  # e.g., bb_mid_20
    re.compile(r"^donchian_high_\d+$"),           # e.g., donchian_high_20
    re.compile(r"^donchian_low_\d+$"),            # e.g., donchian_low_20
    re.compile(r"^donchian_mid_\d+$"),            # e.g., donchian_mid_20
    re.compile(r"^keltner_upper_\d+$"),           # e.g., keltner_upper_20
    re.compile(r"^keltner_lower_\d+$"),           # e.g., keltner_lower_20
    re.compile(r"^mom_\d+$"),                     # raw price delta (close - close.shift)
    re.compile(r"^close_slope_\d+$"),             # raw price slope (delta price / days)
    re.compile(r"^hl_spread$"),                   # raw high - low nominal price spread
    re.compile(r"^co_spread$"),                   # raw close - open nominal price spread
    re.compile(r"^upper_shadow$"),                # raw upper shadow nominal price spread
    re.compile(r"^lower_shadow$"),                # raw lower shadow nominal price spread
]


def is_stationary_feature(col_name: str) -> bool:
    """
    Check whether a column name violates stationarity rules.

    Returns False if col_name represents a raw price level, raw volume level,
    raw nominal MA/ATR/MACD line, or raw price spread.
    Returns True if stationary (ratio, percentage return, bounded oscillator).
    """
    name_lower = col_name.strip().lower()
    if name_lower in PROHIBITED_STATIONARY_COLUMNS:
        return False
    for pat in PROHIBITED_PATTERNS:
        if pat.match(name_lower):
            return False
    return True


def validate_stationarity(feature_cols: List[str], allow_violations: bool = False) -> List[str]:
    """
    Validate that all feature columns in a feature set comply with stationarity rules.

    Parameters
    ----------
    feature_cols : List[str]
        List of candidate feature column names.
    allow_violations : bool, default False
        If True, log warnings and return violations. If False, raise ValueError on any violation.

    Returns
    -------
    List[str]
        List of violating column names found.
    """
    violations = [c for c in feature_cols if not is_stationary_feature(c)]
    if violations:
        msg = f"Stationarity violation detected! {len(violations)} non-stationary columns found: {violations[:10]}"
        if allow_violations:
            logger.warning(msg)
        else:
            raise ValueError(msg)
    return violations


# ==============================================================================
# Feature Set Definitions
# ==============================================================================

# SET A: Minimal Stationary Technicals
SET_A_FEATURES: List[str] = [
    # Multi-horizon percentage & log returns
    "return_1d", "log_return_1d",
    "return_2d", "log_return_2d",
    "return_3d", "log_return_3d",
    "return_5d", "log_return_5d",
    "return_10d", "log_return_10d",
    "return_15d", "log_return_15d",
    "return_21d", "log_return_21d",
    "return_42d", "return_63d",
    # Relative momentum (Rate of Change %)
    "roc_1", "roc_2", "roc_3", "roc_5", "roc_10", "roc_15", "roc_21", "roc_42", "roc_63",
    # Realized volatility (annualized from daily returns)
    "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d", "volatility_120d",
    "volatility_parkinson_20", "volatility_garman_klass_20",
    # ATR% (Normalized ATR) & Percentage Spreads
    "natr_7", "natr_14", "natr_21",
    "hl_spread_pct", "co_spread_pct",
    # Volume Ratios & Money Flow
    "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "vol_ratio_50",
    "cmf_20",
]

# SET B: SET A + Market & Sector Context
SET_B_ADDITIONS: List[str] = [
    # Benchmark returns
    "nifty_50_close_return", "nifty_50_return_5d", "nifty_50_return_21d",
    # Sector index returns
    "nifty_bank_close_return", "nifty_bank_return_5d", "nifty_bank_return_21d",
    # Volatility index context
    "vix_close_return", "vix_change_5d", "vix_level_scaled",
    # Macro context (if available)
    "usdinr_return_1d", "usdinr_return_5d", "crude_return_1d", "crude_return_5d",
    # Relative Strength (RS) vs Benchmark & Sector
    "rs_nifty_5d", "rs_nifty_21d",
    "rs_sector_5d", "rs_sector_21d",
]

SET_B_FEATURES: List[str] = SET_A_FEATURES + SET_B_ADDITIONS

# SET C: SET B + Full Stationary Technicals (Oscillators, Normalized MACD, %B, ADX, MA Distances)
SET_C_ADDITIONS: List[str] = [
    # Selected RSI
    "rsi_7", "rsi_14", "rsi_21", "rsi_28",
    # Normalized MACD Histograms (MACD Hist / Close)
    "macd_hist_12_26_norm", "macd_hist_8_17_norm", "macd_hist_5_35_norm",
    # Bollinger %B and Width
    "bb_pct_b_10", "bb_pct_b_20", "bb_pct_b_50",
    "bb_width_10", "bb_width_20", "bb_width_50",
    # ADX and Directional Indicators
    "adx_14", "plus_di_14", "minus_di_14",
    # Percentage distance to Moving Averages & MA ratios
    "dist_sma_5", "dist_ema_5",
    "dist_sma_10", "dist_ema_10",
    "dist_sma_15", "dist_ema_15",
    "dist_sma_20", "dist_ema_20",
    "dist_sma_30", "dist_ema_30",
    "dist_sma_50", "dist_ema_50",
    "dist_sma_100", "dist_ema_100",
    "dist_sma_150", "dist_ema_150",
    "dist_sma_200", "dist_ema_200",
    "sma_5_20_ratio", "sma_20_50_ratio", "sma_50_200_ratio",
    # Stationary Bounded Oscillators & Channels
    "stoch_k_14", "stoch_d_14", "stoch_k_21", "stoch_d_21",
    "williams_r_7", "williams_r_14", "williams_r_28",
    "cci_14", "cci_20",
    "donchian_pos_10", "donchian_pos_20",
    "keltner_pos_20",
    # Stationary Return Lags
    "return_1d_lag_1", "return_1d_lag_2", "return_1d_lag_3", "return_1d_lag_5",
    "rsi_14_lag_1", "rsi_14_lag_2", "rsi_14_lag_3",
    "volatility_20d_lag_1", "volatility_20d_lag_2",
]

SET_C_FEATURES: List[str] = SET_B_FEATURES + SET_C_ADDITIONS

# SET D: SET C + News & NLP Features (from WS7 / sentiment aggregator)
SET_D_ADDITIONS: List[str] = [
    "sentiment_mean",
    "sentiment_pos_ratio",
    "sentiment_neg_ratio",
    "sentiment_momentum_3d",
    "sentiment_momentum_7d",
    "sentiment_divergence_flag",
    "news_count_log",
]

SET_D_FEATURES: List[str] = SET_C_FEATURES + SET_D_ADDITIONS


def get_feature_set(
    set_name: str,
    df: Optional[pd.DataFrame] = None,
    enforce_stationarity: bool = True,
) -> List[str]:
    """
    Retrieve feature column names for a given ablation set ('A', 'B', 'C', 'D', 'E').
    If a dataframe is provided, filters for columns actually present in df.

    Parameters
    ----------
    set_name : str
        One of 'A', 'B', 'C', 'D', 'E' (case-insensitive).
    df : Optional[pd.DataFrame]
        If provided, intersect requested set with available columns in df.
    enforce_stationarity : bool, default True
        If True, validates stationarity for sets A, B, C, D (raises on violation).

    Returns
    -------
    List[str]
        List of feature column names.
    """
    key = set_name.strip().upper()
    if key == "A":
        features = list(SET_A_FEATURES)
    elif key == "B":
        features = list(SET_B_FEATURES)
    elif key == "C":
        features = list(SET_C_FEATURES)
    elif key == "D":
        features = list(SET_D_FEATURES)
    elif key == "E":
        # Control set: all numeric columns in df excluding identifiers and targets
        if df is None:
            raise ValueError("Dataframe must be provided to determine Set E (all features).")
        exclude_exact = {
            "ticker", "date", "market_regime", "label_start",
            "y_5d", "y_21d", "y_126d",
            "y_5d_raw", "y_21d_raw", "y_126d_raw",
            "y_5d_excess", "y_21d_excess", "y_126d_excess",
            "y_5d_clipped", "y_21d_clipped", "y_126d_clipped",
            "y_5d_ranked", "y_21d_ranked", "y_126d_ranked",
            "label_end_5d", "label_end_21d", "label_end_126d",
            "Target_1D", "Target_5D", "Target_21D", "Target_126D",
            "divergence_desc",
        }
        features = [
            c for c in df.columns
            if c not in exclude_exact and pd.api.types.is_numeric_dtype(df[c])
        ]
        return sorted(features)
    else:
        raise ValueError(f"Unknown feature set '{set_name}'. Expected 'A', 'B', 'C', 'D', or 'E'.")

    if enforce_stationarity and key in {"A", "B", "C", "D"}:
        validate_stationarity(features, allow_violations=False)

    if df is not None:
        features = [c for c in features if c in df.columns]

    return features


def enrich_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive essential stationary features if not already precomputed in dataframe:
      - macd_hist_*_norm = macd_hist_* / close
      - nifty_50_return_5d, nifty_50_return_21d
      - vix_change_5d, vix_level_scaled
      - rs_nifty_5d, rs_nifty_21d
      - rs_sector_5d, rs_sector_21d
      - news_count_log

    Guarantees no future lookahead leakage by using purely backward-looking operations.
    """
    df = df.copy()
    close = df["close"] if "close" in df.columns else df.get("adj_close")
    ticker_col = "ticker" if "ticker" in df.columns else None

    # 1. Normalized MACD Histograms
    for fast, slow in [(12, 26), (8, 17), (5, 35)]:
        hist_col = f"macd_hist_{fast}_{slow}"
        norm_col = f"{hist_col}_norm"
        if hist_col in df.columns and close is not None and norm_col not in df.columns:
            df[norm_col] = df[hist_col] / (close + 1e-10)

    # 2. Multi-day benchmark returns
    if "nifty_50_close" in df.columns:
        nifty = df["nifty_50_close"]
        for p in [5, 21]:
            col = f"nifty_50_return_{p}d"
            if col not in df.columns:
                if ticker_col:
                    df[col] = df.groupby(ticker_col)["nifty_50_close"].pct_change(p)
                else:
                    df[col] = nifty.pct_change(p)

    # 3. Multi-day sector returns (Nifty Bank)
    if "nifty_bank_close" in df.columns:
        bank = df["nifty_bank_close"]
        for p in [5, 21]:
            col = f"nifty_bank_return_{p}d"
            if col not in df.columns:
                if ticker_col:
                    df[col] = df.groupby(ticker_col)["nifty_bank_close"].pct_change(p)
                else:
                    df[col] = bank.pct_change(p)

    # 4. VIX features
    if "vix_close" in df.columns:
        vix = df["vix_close"]
        if "vix_level_scaled" not in df.columns:
            df["vix_level_scaled"] = vix / 100.0
        if "vix_change_5d" not in df.columns:
            if ticker_col:
                df["vix_change_5d"] = df.groupby(ticker_col)["vix_close"].pct_change(5)
            else:
                df["vix_change_5d"] = vix.pct_change(5)

    # 5. Relative Strength (RS) vs Benchmark
    for p in [5, 21]:
        stock_ret_col = f"return_{p}d"
        nifty_ret_col = f"nifty_50_return_{p}d"
        rs_nifty_col = f"rs_nifty_{p}d"
        if stock_ret_col in df.columns and nifty_ret_col in df.columns and rs_nifty_col not in df.columns:
            df[rs_nifty_col] = df[stock_ret_col] - df[nifty_ret_col]

        bank_ret_col = f"nifty_bank_return_{p}d"
        rs_bank_col = f"rs_sector_{p}d"
        if stock_ret_col in df.columns and bank_ret_col in df.columns and rs_bank_col not in df.columns:
            df[rs_bank_col] = df[stock_ret_col] - df[bank_ret_col]

    # 6. News features
    if "news_count" in df.columns and "news_count_log" not in df.columns:
        df["news_count_log"] = np.log1p(df["news_count"].fillna(0))

    return df
