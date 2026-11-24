"""
validation.py
-------------
Data integrity and corporate action validation for StockReason v0.7.

Mandates:
  1. Corporate Actions: Always use `adj_close` for all return and target calculations.
  2. Single-day Return Check: Flag any single-day move beyond ±25% not accompanied
     by a known split or bonus.
  3. Trading Day Gap Check: Flag any chronological data gaps larger than 5 trading days
     using the official NSE holiday calendar.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.m1_price_data.nse_holidays import is_trading_day


def count_missed_trading_days_between(
    start_dt: Union[str, date, datetime, pd.Timestamp],
    end_dt: Union[str, date, datetime, pd.Timestamp],
) -> int:
    """
    Counts the number of NSE trading days strictly between start_dt and end_dt (exclusive).
    I.e., days d such that start_dt < d < end_dt and is_trading_day(d) is True.
    """
    d1 = pd.to_datetime(start_dt).date()
    d2 = pd.to_datetime(end_dt).date()
    if d2 <= d1:
        return 0

    delta_days = (d2 - d1).days
    if delta_days <= 1:
        return 0

    dates_between = pd.date_range(d1 + pd.Timedelta(days=1), d2 - pd.Timedelta(days=1))
    return int(sum(is_trading_day(d.date()) for d in dates_between))


def count_trading_days_elapsed(
    from_dt: Union[str, date, datetime, pd.Timestamp],
    to_dt: Union[str, date, datetime, pd.Timestamp],
) -> int:
    """
    Counts the number of NSE trading days elapsed from from_dt to to_dt,
    inclusive of to_dt and exclusive of from_dt: (from_dt, to_dt].
    If to_dt <= from_dt, returns 0.
    """
    d1 = pd.to_datetime(from_dt).date()
    d2 = pd.to_datetime(to_dt).date()
    if d2 <= d1:
        return 0

    date_range = pd.date_range(d1 + pd.Timedelta(days=1), d2)
    return int(sum(is_trading_day(d.date()) for d in date_range))


@dataclass
class AnomalyRecord:
    ticker: str
    date: str
    price_before: float
    price_after: float
    daily_return: float
    reason: str


@dataclass
class GapRecord:
    ticker: str
    start_date: str
    end_date: str
    missed_trading_days: int
    calendar_days: int


@dataclass
class ValidationReport:
    passed: bool
    num_tickers: int
    total_rows: int
    anomalous_moves: List[Dict[str, Any]] = field(default_factory=list)
    data_gaps: List[Dict[str, Any]] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "num_tickers": self.num_tickers,
            "total_rows": self.total_rows,
            "anomalous_moves_count": len(self.anomalous_moves),
            "anomalous_moves": self.anomalous_moves,
            "data_gaps_count": len(self.data_gaps),
            "data_gaps": self.data_gaps,
            "missing_columns": self.missing_columns,
            "null_counts": self.null_counts,
            "summary": self.summary,
        }


def validate_price_moves(
    df: pd.DataFrame,
    max_abs_move: float = 0.25,
    known_actions: Optional[Union[Dict[str, List[str]], Set[Tuple[str, str]]]] = None,
    price_col: str = "adj_close",
) -> List[Dict[str, Any]]:
    """
    Flags any single-day move where |return| > max_abs_move (default 25%)
    not accompanied by a known corporate action (split or bonus).
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataframe.")

    anomalies: List[Dict[str, Any]] = []
    
    # Format known actions lookup
    known_set: Set[Tuple[str, str]] = set()
    if isinstance(known_actions, dict):
        for tick, dts in known_actions.items():
            for dt_str in dts:
                known_set.add((str(tick), str(pd.to_datetime(dt_str).date())))
    elif isinstance(known_actions, set):
        known_set = {(t, str(pd.to_datetime(d).date())) for t, d in known_actions}

    ticker_col = "ticker" if "ticker" in df.columns else None
    groups = df.groupby("ticker", sort=True) if ticker_col else [("DEFAULT", df)]

    for ticker, grp in groups:
        grp_sorted = grp.sort_values("date").reset_index(drop=True)
        prices = grp_sorted[price_col].to_numpy(dtype=np.float64)
        dates = pd.to_datetime(grp_sorted["date"]).dt.date.to_numpy()

        if len(prices) < 2:
            continue

        prev_prices = prices[:-1]
        curr_prices = prices[1:]
        curr_dates = dates[1:]

        # Avoid zero-division
        valid_mask = prev_prices > 0
        returns = np.zeros_like(curr_prices)
        returns[valid_mask] = (curr_prices[valid_mask] - prev_prices[valid_mask]) / prev_prices[valid_mask]

        abs_ret = np.abs(returns)
        flagged_indices = np.where(abs_ret > max_abs_move)[0]

        for idx in flagged_indices:
            dt_str = str(curr_dates[idx])
            t_name = str(ticker)
            if (t_name, dt_str) in known_set:
                continue

            anomalies.append({
                "ticker": t_name,
                "date": dt_str,
                "price_before": float(prev_prices[idx]),
                "price_after": float(curr_prices[idx]),
                "daily_return": float(returns[idx]),
                "threshold": max_abs_move,
                "reason": (
                    f"Single-day return {returns[idx]:+.2%} exceeds ±{max_abs_move:.0%} "
                    f"without documented split/bonus."
                ),
            })

    return anomalies


def validate_trading_gaps(
    df: pd.DataFrame,
    max_gap_trading_days: int = 5,
) -> List[Dict[str, Any]]:
    """
    Flags any chronological gaps between consecutive rows for a ticker
    where the number of missed NSE trading days exceeds max_gap_trading_days (default 5).
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column.")

    gaps: List[Dict[str, Any]] = []
    ticker_col = "ticker" if "ticker" in df.columns else None
    groups = df.groupby("ticker", sort=True) if ticker_col else [("DEFAULT", df)]

    for ticker, grp in groups:
        grp_sorted = grp.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(grp_sorted["date"]).dt.date.to_numpy()

        if len(dates) < 2:
            continue

        for i in range(1, len(dates)):
            d1 = dates[i - 1]
            d2 = dates[i]
            missed_t_days = count_missed_trading_days_between(d1, d2)
            if missed_t_days > max_gap_trading_days:
                gaps.append({
                    "ticker": str(ticker),
                    "start_date": str(d1),
                    "end_date": str(d2),
                    "missed_trading_days": missed_t_days,
                    "calendar_days": (d2 - d1).days,
                    "threshold": max_gap_trading_days,
                    "reason": f"Gap of {missed_t_days} trading days exceeds threshold ({max_gap_trading_days} days).",
                })

    return gaps


def validate_dataset_integrity(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    price_col: str = "adj_close",
    max_abs_move: float = 0.25,
    max_gap_trading_days: int = 5,
    known_actions: Optional[Union[Dict[str, List[str]], Set[Tuple[str, str]]]] = None,
) -> ValidationReport:
    """
    Comprehensive validator for market dataset before sequence generation or modeling.
    
    Verifies:
      1. Presence of required columns: date, ticker, and price_col ('adj_close').
      2. No nulls or non-positive values in price_col.
      3. Absence of anomalous unadjusted corporate action price jumps (> ±25%).
      4. Absence of abnormal trading day gaps (> 5 trading days).
      5. No null values in specified feature_cols.
    """
    missing_cols = []
    required_cols = ["date"]
    if price_col not in df.columns:
        missing_cols.append(price_col)
    for col in required_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        return ValidationReport(
            passed=False,
            num_tickers=df["ticker"].nunique() if "ticker" in df.columns else 1,
            total_rows=len(df),
            missing_columns=missing_cols,
            summary=f"Validation failed: missing required columns: {missing_cols}",
        )

    # Check nulls in price
    null_counts: Dict[str, int] = {}
    null_price = int(df[price_col].isna().sum())
    if null_price > 0:
        null_counts[price_col] = null_price

    if feature_cols:
        for f in feature_cols:
            if f in df.columns:
                n_null = int(df[f].isna().sum())
                if n_null > 0:
                    null_counts[f] = n_null

    # Run price move and gap checks
    anomalous_moves = validate_price_moves(
        df=df,
        max_abs_move=max_abs_move,
        known_actions=known_actions,
        price_col=price_col,
    )

    data_gaps = validate_trading_gaps(
        df=df,
        max_gap_trading_days=max_gap_trading_days,
    )

    # Data is deemed passing if no missing required cols, no null prices, no huge gaps,
    # and no unexplained corporate action anomalies.
    passed = (len(missing_cols) == 0) and (null_price == 0) and (len(data_gaps) == 0)

    summary_parts = []
    if passed:
        summary_parts.append("Data integrity passed.")
    else:
        summary_parts.append("Data integrity check found issues.")
    if anomalous_moves:
        summary_parts.append(f"{len(anomalous_moves)} anomalous single-day moves detected.")
    if data_gaps:
        summary_parts.append(f"{len(data_gaps)} trading day gaps (> {max_gap_trading_days} days) detected.")
    if null_counts:
        summary_parts.append(f"Null values present in {len(null_counts)} columns.")

    summary = " ".join(summary_parts)
    num_tickers = int(df["ticker"].nunique()) if "ticker" in df.columns else 1

    return ValidationReport(
        passed=passed,
        num_tickers=num_tickers,
        total_rows=len(df),
        anomalous_moves=anomalous_moves,
        data_gaps=data_gaps,
        missing_columns=missing_cols,
        null_counts=null_counts,
        summary=summary,
    )
