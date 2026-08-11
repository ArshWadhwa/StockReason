import logging
from typing import Dict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw stock OHLCV DataFrame.
    Handles:
    - Normalizing dates to YYYY-MM-DD
    - Dropping duplicate (date, ticker) records
    - Replacing zero/negative price values with NaNs and forward/backward filling
    - Imputing missing volume with 0
    - Enforcing logical price boundaries (high >= low, high >= open, high >= close)
    """
    if df.empty:
        return df

    cleaned = df.copy()

    # 1. Normalize dates
    cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.normalize()

    # 2. Deduplicate
    initial_len = len(cleaned)
    if "ticker" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["date", "ticker"], keep="last")
    elif "symbol" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["date", "symbol"], keep="last")
    else:
        cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")

    dedup_dropped = initial_len - len(cleaned)
    if dedup_dropped > 0:
        logger.info(f"Removed {dedup_dropped} duplicate rows.")

    # 3. Clean price zero/negative anomalies
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in cleaned.columns]
    for col in price_cols:
        cleaned[col] = cleaned[col].apply(lambda x: np.nan if x is not None and x <= 0 else x)

    # Impute missing values per ticker group
    if "ticker" in cleaned.columns:
        cleaned[price_cols] = cleaned.groupby("ticker")[price_cols].ffill().bfill()
    else:
        cleaned[price_cols] = cleaned[price_cols].ffill().bfill()

    # 4. Enforce high/low logical boundaries
    if all(c in cleaned.columns for c in ["high", "low", "open", "close"]):
        cleaned["high"] = cleaned[["high", "open", "close"]].max(axis=1)
        cleaned["low"] = cleaned[["low", "open", "close"]].min(axis=1)

    # 5. Volume handling
    if "volume" in cleaned.columns:
        cleaned["volume"] = cleaned["volume"].fillna(0).astype("int64")

    return cleaned.sort_values(["ticker", "date"] if "ticker" in cleaned.columns else ["date"]).reset_index(drop=True)


def align_and_clean_dataset(
    stocks_df: pd.DataFrame,
    indices_dict: Dict[str, pd.DataFrame],
    macro_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge and align stock OHLCV data with market indices (Nifty 50, Nifty Bank, Nifty IT, India VIX)
    and macroeconomic indicators (USD/INR, Crude Oil) into a unified time-series DataFrame.
    """
    logger.info("Aligning stock prices with market context (Indices & Macro data)...")

    cleaned_stocks = clean_price_data(stocks_df)
    if cleaned_stocks.empty:
        return pd.DataFrame()

    merged_df = cleaned_stocks.copy()

    # Merge index levels
    index_name_col_map = {
        "nifty_50": "nifty_50_close",
        "nifty_bank": "nifty_bank_close",
        "nifty_it": "nifty_it_close",
        "india_vix": "vix_close"
    }

    for idx_key, idx_df in indices_dict.items():
        if idx_df.empty:
            continue
        cleaned_idx = clean_price_data(idx_df)

        col_name = index_name_col_map.get(idx_key, f"{idx_key}_close")
        sub_idx = cleaned_idx[["date", "close"]].rename(columns={"close": col_name})
        sub_idx[f"{col_name}_return"] = sub_idx[col_name].pct_change()

        merged_df = pd.merge(merged_df, sub_idx, on="date", how="left")

    # Merge macro indicators
    macro_name_col_map = {
        "usd_inr": "usd_inr_close",
        "crude_oil": "crude_oil_close"
    }

    for macro_key, macro_df in macro_dict.items():
        if macro_df.empty:
            continue
        cleaned_macro = clean_price_data(macro_df)

        col_name = macro_name_col_map.get(macro_key, f"{macro_key}_close")
        sub_macro = cleaned_macro[["date", "close"]].rename(columns={"close": col_name})
        sub_macro[f"{col_name}_return"] = sub_macro[col_name].pct_change()

        merged_df = pd.merge(merged_df, sub_macro, on="date", how="left")

    # Forward-fill missing market/macro context values per ticker
    context_cols = [c for c in merged_df.columns if c not in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    if context_cols:
        merged_df[context_cols] = merged_df.groupby("ticker")[context_cols].ffill().bfill()

    return merged_df.sort_values(["ticker", "date"]).reset_index(drop=True)
