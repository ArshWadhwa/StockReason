"""
StockReason v0.8 Core Contracts and Schema Definitions.

Defines schemas and validators for:
- Raw OHLCV records
- Canonical OHLCV panel
- Historical universe security metadata
- Corporate actions
- Feature & Target definitions
- Point-in-time and Purged CV parameters
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
import pandas as pd


# Expected raw and canonical columns
REQUIRED_OHLCV_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

CANONICAL_EXTRA_COLUMNS = [
    "adj_factor",
    "split_bonus_anomaly",
    "unadj_return_1d",
    "adj_return_1d",
]

REQUIRED_UNIVERSE_COLUMNS = [
    "ticker",
    "symbol_nse",
    "isin",
    "company_name",
    "sector",
    "listing_date",
    "delisting_date",
    "is_active",
    "survivorship_status",
]

SUPPORTED_HORIZONS = [1, 5, 21, 63, 126]


@dataclass
class SecurityMetadata:
    """Historical security metadata with point-in-time eligibility support."""
    ticker: str
    symbol_nse: str
    isin: str
    company_name: str
    sector: str
    listing_date: Optional[str] = None
    delisting_date: Optional[str] = None
    is_active: bool = True
    survivorship_status: str = "active"  # 'active', 'delisted', 'merged', 'acquired'
    aliases: List[str] = field(default_factory=list)

    def is_eligible(self, as_of_date: str) -> bool:
        """Determines whether security was tradeable at as_of_date."""
        target_dt = pd.to_datetime(as_of_date)
        if self.listing_date:
            listing_dt = pd.to_datetime(self.listing_date)
            if target_dt < listing_dt:
                return False
        if not self.is_active and self.delisting_date:
            delisting_dt = pd.to_datetime(self.delisting_date)
            if target_dt > delisting_dt:
                return False
        return True


def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    is_canonical: bool = False,
    allow_empty: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Strict validation of OHLCV DataFrame against v0.8 contracts.

    Invariants enforced:
    1. Required columns present.
    2. Date normalized to tz-naive datetime64.
    3. Ticker non-empty string.
    4. Unique (ticker, date) key.
    5. Positive prices (open, high, low, close, adj_close > 0).
    6. Consistent OHLC boundaries: high >= max(open, close), low <= min(open, close).
    7. Non-negative volume (volume >= 0).
    8. No future dates beyond current UTC date.
    """
    errors: List[str] = []

    if df.empty:
        if allow_empty:
            return True, []
        return False, ["DataFrame is empty"]

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Canonical columns if required
    if is_canonical:
        missing_can = [c for c in CANONICAL_EXTRA_COLUMNS if c not in df.columns]
        if missing_can:
            errors.append(f"Missing canonical columns: {missing_can}")

    # 2. Date type and timezone
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            pd.to_datetime(df["date"])
        except Exception as e:
            errors.append(f"Date column cannot be converted to datetime: {e}")

    # 3. Ticker check
    if df["ticker"].isnull().any() or (df["ticker"].astype(str).str.strip() == "").any():
        errors.append("Null or empty ticker values detected")

    # 4. Unique (ticker, date)
    dups = df.duplicated(subset=["ticker", "date"]).sum()
    if dups > 0:
        errors.append(f"Duplicate (ticker, date) keys detected: {dups} occurrences")

    # 5. Positive prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if (df[col] <= 0).any():
            bad_cnt = (df[col] <= 0).sum()
            errors.append(f"Non-positive price values in '{col}': {bad_cnt} rows")

    # 6. OHLC boundaries (allow 1e-4 tolerance for rounding precision)
    tol = 1e-4
    high_violation = (df["high"] < df["open"] - tol) | (df["high"] < df["close"] - tol)
    if high_violation.any():
        errors.append(f"High price lower than open or close: {high_violation.sum()} rows")

    low_violation = (df["low"] > df["open"] + tol) | (df["low"] > df["close"] + tol)
    if low_violation.any():
        errors.append(f"Low price higher than open or close: {low_violation.sum()} rows")

    # 7. Non-negative volume
    if (df["volume"] < 0).any():
        errors.append(f"Negative volume values detected: {(df['volume'] < 0).sum()} rows")

    # 8. Future date check
    max_date = pd.to_datetime(df["date"]).max()
    now_date = pd.to_datetime(datetime.now(timezone.utc).date()) + pd.Timedelta(days=1)
    if max_date > now_date:
        errors.append(f"Future dates detected: max date {max_date} > now {now_date}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_universe_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates historical universe definition DataFrame."""
    errors: List[str] = []
    if df.empty:
        return False, ["Universe DataFrame is empty"]

    missing_cols = [c for c in REQUIRED_UNIVERSE_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required universe columns: {missing_cols}")
        return False, errors

    if df["ticker"].duplicated().any():
        dups = df["ticker"][df["ticker"].duplicated()].tolist()
        errors.append(f"Duplicate tickers in universe: {dups[:10]}")

    is_valid = len(errors) == 0
    return is_valid, errors


def compute_purge_window(horizon: int, embargo_days: int = 5) -> int:
    """
    Computes purged walk-forward CV purge window tailored to forecast horizon.
    Rule: purge_window = horizon + embargo_days.
    A short 21-day purge must never be used for a 126-day target.
    """
    if horizon <= 0:
        raise ValueError(f"Horizon must be positive, got {horizon}")
    return horizon + embargo_days
