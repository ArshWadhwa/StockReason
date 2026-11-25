"""Verified Exchange Trading Calendar & Session Utilities."""

from datetime import datetime, date, timedelta
from typing import Optional, Any, List, Set, Tuple
import pandas as pd

# Major Fixed Indian National / NSE Exchange Holidays (Month, Day)
NSE_FIXED_HOLIDAYS: Set[Tuple[int, int]] = {
    (1, 26),   # Republic Day
    (4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    (5, 1),    # Maharashtra Day
    (8, 15),   # Independence Day
    (10, 2),   # Mahatma Gandhi Jayanti
    (12, 25),  # Christmas
}


def is_trading_day(dt: date) -> bool:
    """Check if a date is a weekday and not a fixed recognized NSE exchange holiday."""
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    if (dt.month, dt.day) in NSE_FIXED_HOLIDAYS:
        return False
    return True


def get_target_trading_date(
    start_date: Optional[Any] = None,
    horizon_days: int = 21,
) -> str:
    """
    Computes verified forward target exchange trading date skipping weekends
    and major recognized National Stock Exchange (NSE) trading holidays.
    """
    if start_date is None:
        curr = datetime.now().date()
    elif isinstance(start_date, str):
        curr = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    elif isinstance(start_date, (datetime, pd.Timestamp)):
        curr = start_date.date()
    else:
        curr = start_date

    days_added = 0
    candidate = curr

    while days_added < horizon_days:
        candidate += timedelta(days=1)
        if not is_trading_day(candidate):
            continue
        days_added += 1

    return candidate.strftime("%Y-%m-%d")


def compute_trading_calendar_sessions(
    df: pd.DataFrame,
    date_col: str = "date",
) -> List[pd.Timestamp]:
    """Extracts unique sorted exchange trading dates from a dataset."""
    dates = pd.to_datetime(df[date_col].unique())
    return sorted(dates)
