"""Data Cleaning & Deduplication Module."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


def clean_market_dataset(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Cleans market data:
    1. Ensures datetime conversion and sorting by (symbol, date).
    2. Drops exact duplicates on (symbol, date).
    3. Drops observations where closing price <= 0.
    4. Computes data cleaning summary.
    """
    out = df.copy()
    orig_count = len(out)

    if date_col in out.columns and not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col])

    # Deduplication
    out = out.drop_duplicates(subset=[symbol_col, date_col], keep="last")
    dedup_dropped = orig_count - len(out)

    # Positive price enforcement
    if "close" in out.columns:
        valid_price_mask = (out["close"] > 0)
        invalid_price_count = int((~valid_price_mask).sum())
        out = out[valid_price_mask]
    else:
        invalid_price_count = 0

    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)

    summary = {
        "original_rows": orig_count,
        "final_rows": len(out),
        "dedup_dropped": dedup_dropped,
        "invalid_price_dropped": invalid_price_count,
    }
    return out, summary
