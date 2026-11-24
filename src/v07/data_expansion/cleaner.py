"""
Data cleaning, anomaly detection, calendar alignment, and liquidity/history filtering.
Ensures zero data leakage and strict alignment to NSE trading days.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ws6.cleaner")

# Thresholds per contract
MIN_TRADING_DAYS_DEFAULT = 750
MIN_MEDIAN_TRADED_VALUE_INR = 50_000_000.0  # ₹5 crore/day
SPLIT_ANOMALY_THRESHOLD = 0.25  # ±25% jump


def detect_split_bonus_anomalies(
    df: pd.DataFrame,
    threshold: float = SPLIT_ANOMALY_THRESHOLD,
) -> pd.DataFrame:
    """
    Detects split/bonus anomalies and unadjusted price jumps.

    Flags rows where:
    1. Unadjusted close price jumps by >= threshold (±25%) while adjusted close
       does not jump proportionally (ratio anomaly):
       `ratio_diff = abs((close[t] / close[t-1]) / (adj_close[t] / adj_close[t-1]) - 1.0) >= threshold`
    2. Or unadjusted close jump is >= threshold and absolute difference between
       unadjusted return and adjusted return exceeds 15%.

    Returns copy of df with:
    - 'split_bonus_anomaly': bool flag
    - 'unadj_return_1d': 1-day unadjusted return
    - 'adj_return_1d': 1-day adjusted return
    - 'split_ratio_diff': divergence between unadj and adj relative returns
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["split_bonus_anomaly"] = False
        return df_copy

    df_copy = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    def _calc_anomalies(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        close = g["close"]
        adj_close = g.get("adj_close", close)

        prev_close = close.shift(1)
        prev_adj = adj_close.shift(1)

        unadj_ret = (close - prev_close) / prev_close
        adj_ret = (adj_close - prev_adj) / prev_adj

        # Ratio of unadjusted step to adjusted step
        unadj_ratio = close / (prev_close + 1e-9)
        adj_ratio = adj_close / (prev_adj + 1e-9)
        ratio_diff = (unadj_ratio / (adj_ratio + 1e-9)) - 1.0

        # Anomaly condition: ratio difference >= threshold, or unadjusted jump >= threshold
        # with significant gap from adjusted return
        is_ratio_anomaly = ratio_diff.abs() >= threshold
        is_unadj_jump_anomaly = (unadj_ret.abs() >= threshold) & ((unadj_ret - adj_ret).abs() >= 0.15)

        anomaly = is_ratio_anomaly | is_unadj_jump_anomaly
        # The very first row per group cannot have a step anomaly
        anomaly.iloc[0] = False

        g["unadj_return_1d"] = unadj_ret
        g["adj_return_1d"] = adj_ret
        g["split_ratio_diff"] = ratio_diff
        g["split_bonus_anomaly"] = anomaly.fillna(False)
        return g

    if "ticker" in df_copy.columns:
        result_dfs = [_calc_anomalies(group) for _, group in df_copy.groupby("ticker")]
        result = pd.concat(result_dfs, ignore_index=True) if result_dfs else df_copy
    else:
        result = _calc_anomalies(df_copy)

    anomalies_count = result["split_bonus_anomaly"].sum()
    if anomalies_count > 0:
        logger.info(f"Detected {anomalies_count} split/bonus anomaly rows across panel.")
    return result


def detect_zero_volume_runs(
    df: pd.DataFrame,
    min_consecutive: int = 5,
) -> pd.DataFrame:
    """
    Detects sustained runs of zero trading volume (halted trading or illiquid periods).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'volume' and optionally 'ticker'.
    min_consecutive : int
        Minimum consecutive days of zero volume to flag.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'zero_vol_run': bool and 'zero_vol_streak': int.
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["zero_vol_run"] = False
        df_copy["zero_vol_streak"] = 0
        return df_copy

    df_copy = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    def _calc_zero_runs(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        is_zero = (g["volume"].fillna(0) <= 0).astype(int)

        # Count consecutive runs
        streak = (is_zero * (is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumcount() + 1))
        # Flag streaks that reach or exceed min_consecutive
        g["zero_vol_streak"] = streak
        g["zero_vol_run"] = streak >= min_consecutive
        return g

    if "ticker" in df_copy.columns:
        res_dfs = [_calc_zero_runs(group) for _, group in df_copy.groupby("ticker")]
        return pd.concat(res_dfs, ignore_index=True) if res_dfs else df_copy
    return _calc_zero_runs(df_copy)


def filter_minimum_history(
    df: pd.DataFrame,
    min_trading_days: int = MIN_TRADING_DAYS_DEFAULT,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Filters out tickers with fewer than `min_trading_days` of history.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        - Cleaned DataFrame containing only retained tickers
        - Dictionary of excluded tickers with stats and reason
    """
    if df.empty or "ticker" not in df.columns:
        return df, {}

    stats = df.groupby("ticker")["date"].agg(rows="count", first_date="min", last_date="max").reset_index()

    retained_tickers = set(stats[stats["rows"] >= min_trading_days]["ticker"])
    excluded_info = {}

    for _, row in stats[stats["rows"] < min_trading_days].iterrows():
        t = row["ticker"]
        excluded_info[t] = {
            "rows": int(row["rows"]),
            "first_date": str(row["first_date"].date()) if pd.notnull(row["first_date"]) else None,
            "last_date": str(row["last_date"].date()) if pd.notnull(row["last_date"]) else None,
            "reason": f"insufficient_history (<{min_trading_days} trading days; actual={row['rows']})",
        }

    filtered_df = df[df["ticker"].isin(retained_tickers)].reset_index(drop=True)
    logger.info(
        f"History filter (>= {min_trading_days} days): retained {len(retained_tickers)} tickers, "
        f"excluded {len(excluded_info)} tickers."
    )
    return filtered_df, excluded_info


def calculate_median_traded_value(
    df: pd.DataFrame,
    window: int = 60,
) -> Dict[str, float]:
    """
    Calculates the median daily traded value (close * volume) in INR over the trailing `window` days
    for each ticker.
    """
    if df.empty or "ticker" not in df.columns:
        return {}

    medians = {}
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date")
        traded_val = g["close"] * g["volume"]
        trailing = traded_val.tail(window)
        med = float(trailing.median()) if not trailing.empty else 0.0
        medians[ticker] = med if not np.isnan(med) else 0.0

    return medians


def filter_liquidity(
    df: pd.DataFrame,
    min_median_value_inr: float = MIN_MEDIAN_TRADED_VALUE_INR,
    window: int = 60,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Filters out tickers whose trailing 60-day median daily traded value is below ₹5 crore/day.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        - Cleaned DataFrame with liquid tickers only
        - Dictionary of excluded tickers with reason and median traded value
    """
    if df.empty or "ticker" not in df.columns:
        return df, {}

    medians = calculate_median_traded_value(df, window=window)
    retained_tickers = set()
    excluded_info = {}

    for ticker, med in medians.items():
        if med >= min_median_value_inr:
            retained_tickers.add(ticker)
        else:
            val_cr = med / 10_000_000.0
            min_cr = min_median_value_inr / 10_000_000.0
            excluded_info[ticker] = {
                "median_traded_value_inr": med,
                "median_traded_value_cr": val_cr,
                "reason": f"low_liquidity (trailing {window}d median traded value = ₹{val_cr:.2f} Cr < ₹{min_cr:.1f} Cr/day)",
            }

    filtered_df = df[df["ticker"].isin(retained_tickers)].reset_index(drop=True)
    logger.info(
        f"Liquidity filter (>= ₹{min_median_value_inr/1e7:.1f} Cr/day): retained {len(retained_tickers)}, "
        f"excluded {len(excluded_info)}."
    )
    return filtered_df, excluded_info


def align_to_nse_calendar(
    stocks_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    calendar_dates: Optional[pd.DatetimeIndex] = None,
    max_ffill_days: int = 5,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Strictly aligns stocks and macro data to valid NSE trading calendar dates.

    Guarantees:
    1. Forward-fill ONLY (strictly NO future data leakage).
    2. Macro indicators are reindexed to the exact NSE trading dates and forward-filled.
    3. If calendar_dates is not passed, uses sorted unique dates of stocks_df as calendar.

    Parameters
    ----------
    stocks_df : pd.DataFrame
        Stock OHLCV panel containing ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume'].
    macro_df : Optional[pd.DataFrame]
        Macro panel or DataFrame with date column and macro columns (e.g. USDINR, Crude).
    calendar_dates : Optional[pd.DatetimeIndex]
        Explicit valid NSE trading dates (e.g. from ^NSEI).
    max_ffill_days : int
        Maximum number of consecutive trading days to forward-fill missing series.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        Aligned stocks panel and aligned macro panel.
    """
    if stocks_df.empty:
        return stocks_df, macro_df

    # Establish calendar
    if calendar_dates is None:
        calendar_dates = pd.DatetimeIndex(sorted(stocks_df["date"].drop_duplicates()))
    else:
        calendar_dates = pd.DatetimeIndex(sorted(pd.Series(calendar_dates).drop_duplicates()))

    # Reindex each stock to the NSE calendar
    aligned_stock_list = []
    for ticker, group in stocks_df.groupby("ticker"):
        g = group.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        # Keep only dates within the stock's actual trading window (first traded date to last)
        min_date = g["date"].min()
        max_date = g["date"].max()
        ticker_calendar = calendar_dates[(calendar_dates >= min_date) & (calendar_dates <= max_date)]

        g_indexed = g.set_index("date").reindex(ticker_calendar)
        g_indexed["ticker"] = ticker

        # Forward fill prices strictly (no bfill!)
        price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in g_indexed.columns]
        g_indexed[price_cols] = g_indexed[price_cols].ffill(limit=max_ffill_days)
        # Missing volume is 0
        if "volume" in g_indexed.columns:
            g_indexed["volume"] = g_indexed["volume"].fillna(0.0)

        aligned_stock_list.append(g_indexed.reset_index().rename(columns={"index": "date"}))

    aligned_stocks = pd.concat(aligned_stock_list, ignore_index=True)
    aligned_stocks["date"] = pd.to_datetime(aligned_stocks["date"]).dt.tz_localize(None).dt.normalize()

    # Align macro DataFrame
    aligned_macro = None
    if macro_df is not None and not macro_df.empty:
        m = macro_df.copy()
        m["date"] = pd.to_datetime(m["date"]).dt.tz_localize(None).dt.normalize()
        m = m.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        m_indexed = m.set_index("date").reindex(calendar_dates)
        # Strict forward fill only to prevent any future lookahead!
        feature_cols = [c for c in m_indexed.columns if c != "date"]
        m_indexed[feature_cols] = m_indexed[feature_cols].ffill(limit=max_ffill_days)

        aligned_macro = m_indexed.reset_index().rename(columns={"index": "date"})

    return aligned_stocks, aligned_macro


def enforce_price_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces physical price boundaries:
    - high >= low
    - high >= open
    - high >= close
    - low <= open
    - low <= close
    - open, high, low, close > 0
    Adjusts anomalies by clamping high = max(high, open, close) and low = min(low, open, close).
    """
    if df.empty:
        return df

    cleaned = df.copy()
    # Replace negative or zero prices with NaN
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in cleaned.columns]
    for col in price_cols:
        cleaned.loc[cleaned[col] <= 0, col] = np.nan

    # Forward fill valid prices per ticker
    if "ticker" in cleaned.columns:
        cleaned[price_cols] = cleaned.groupby("ticker")[price_cols].ffill()
    else:
        cleaned[price_cols] = cleaned[price_cols].ffill()

    cleaned = cleaned.dropna(subset=["close"]).copy()

    if all(c in cleaned.columns for c in ["open", "high", "low", "close"]):
        cleaned["high"] = np.maximum(cleaned["high"], np.maximum(cleaned["open"], cleaned["close"]))
        cleaned["low"] = np.minimum(cleaned["low"], np.minimum(cleaned["open"], cleaned["close"]))

    return cleaned


def clean_universe_panel(
    raw_panel: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    min_trading_days: int = MIN_TRADING_DAYS_DEFAULT,
    min_median_value_inr: float = MIN_MEDIAN_TRADED_VALUE_INR,
    calendar_benchmark_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Full end-to-end cleaning pipeline for WS6 data expansion:
    1. Deduplication and price boundary enforcement
    2. Split/bonus anomaly detection
    3. Zero-volume run detection
    4. Minimum history filter (>= 750 days)
    5. Liquidity filter (>= ₹5 crore/day)
    6. Alignment to NSE calendar (with zero-leakage forward fill for macro)

    Returns:
    - Cleaned panel DataFrame
    - Cleaned macro DataFrame
    - Audit metadata dict
    """
    logger.info(f"Starting panel cleaning for {len(raw_panel)} raw rows...")

    # 1. Price boundaries & deduplication
    panel = raw_panel.drop_duplicates(subset=["date", "ticker"], keep="last").copy()
    panel = enforce_price_boundaries(panel)

    # 2. Split anomaly detection
    panel = detect_split_bonus_anomalies(panel)

    # 3. Zero volume runs
    panel = detect_zero_volume_runs(panel)

    # 4. Minimum history filter
    panel_hist, hist_excluded = filter_minimum_history(panel, min_trading_days=min_trading_days)

    # 5. Liquidity filter
    panel_liq, liq_excluded = filter_liquidity(panel_hist, min_median_value_inr=min_median_value_inr)

    # 6. Alignment to NSE calendar
    calendar_dates = None
    if calendar_benchmark_df is not None and not calendar_benchmark_df.empty:
        calendar_dates = pd.DatetimeIndex(sorted(calendar_benchmark_df["date"].drop_duplicates()))

    aligned_panel, aligned_macro = align_to_nse_calendar(
        stocks_df=panel_liq,
        macro_df=macro_df,
        calendar_dates=calendar_dates,
    )

    all_exclusions = {**hist_excluded, **liq_excluded}
    audit = {
        "raw_rows": len(raw_panel),
        "cleaned_rows": len(aligned_panel),
        "raw_tickers": int(raw_panel["ticker"].nunique()) if "ticker" in raw_panel.columns else 0,
        "retained_tickers": int(aligned_panel["ticker"].nunique()) if "ticker" in aligned_panel.columns else 0,
        "split_anomalies_flagged": int(aligned_panel["split_bonus_anomaly"].sum()) if "split_bonus_anomaly" in aligned_panel.columns else 0,
        "zero_volume_streaks_flagged": int(aligned_panel["zero_vol_run"].sum()) if "zero_vol_run" in aligned_panel.columns else 0,
        "excluded_tickers": all_exclusions,
    }

    logger.info(
        f"Cleaning complete. Retained {audit['retained_tickers']} tickers ({audit['cleaned_rows']} rows). "
        f"Excluded {len(all_exclusions)} tickers."
    )
    return aligned_panel, aligned_macro, audit
