"""
StockReason v0.7 - Module: Point-in-Time News Event Store
Persists news events in Parquet format with strict point-in-time trading day assignment,
preventing any look-ahead bias across Indian market hours, weekends, and holidays.
"""
import os
import hashlib
from datetime import datetime, date, time as dtime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set, Callable
import pandas as pd
import numpy as np

from src.m1_price_data.nse_holidays import is_trading_day
from src.v07.news.dedup import normalize_title, cluster_headlines
from src.v07.news.classifier import classify_headline

# IST Timezone (UTC + 5:30)
IST_TZ = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_TIME = dtime(15, 30)

DEFAULT_STORE_PATH = Path("data/v0.7/news/events.parquet")

EVENT_SCHEMA_COLUMNS = [
    "event_id",
    "ticker",
    "tickers",
    "title",
    "source",
    "url",
    "published_at_utc",
    "first_seen_at_utc",
    "available_at",
    "trading_date_assigned",
    "finbert_pos",
    "finbert_neu",
    "finbert_neg",
    "sentiment_score",
    "confidence",
    "event_type",
    "dedup_cluster_id",
    "backfilled",
]


def _ensure_datetime_utc(dt_val: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Converts a string, Timestamp, or datetime to a timezone-aware UTC datetime."""
    if isinstance(dt_val, str):
        # Handle trailing Z or offsets
        dt_val = dt_val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_val)
    elif isinstance(dt_val, pd.Timestamp):
        dt = dt_val.to_pydatetime()
    else:
        dt = dt_val

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def assign_trading_date(
    available_at_ist: Union[datetime, str],
    is_trading_day_fn: Optional[Callable[[Union[date, datetime]], bool]] = None,
    valid_trading_dates: Optional[Set[str]] = None,
) -> str:
    """
    Strict Point-in-Time Assignment Rule:
    - An article is usable on trading date d ONLY if available_at <= close of d (15:30 IST).
    - If available_at > 15:30 IST, it rolls to the next trading day.
    - Weekend or holiday articles roll forward to the next trading day.
    Returns: 'YYYY-MM-DD'
    """
    if isinstance(available_at_ist, str):
        available_at_ist = datetime.fromisoformat(available_at_ist)

    # Ensure in IST
    if available_at_ist.tzinfo is None:
        dt_ist = available_at_ist.replace(tzinfo=IST_TZ)
    else:
        dt_ist = available_at_ist.astimezone(IST_TZ)

    check_fn = is_trading_day_fn or is_trading_day

    # Rule: If available after market close (15:30 IST), starts checking next day
    if dt_ist.time() > MARKET_CLOSE_TIME:
        curr_date = dt_ist.date() + timedelta(days=1)
    else:
        curr_date = dt_ist.date()

    # Roll forward until reaching a valid trading day
    while True:
        date_str = curr_date.strftime("%Y-%m-%d")
        if valid_trading_dates is not None:
            if date_str in valid_trading_dates:
                return date_str
        elif check_fn(curr_date):
            return date_str
        curr_date += timedelta(days=1)


def generate_event_id(title: str, source: str) -> str:
    """Generates a deterministic 32-character SHA-256 hex digest for an event."""
    norm = normalize_title(title)
    src = (source or "unknown").strip().lower()
    return hashlib.sha256(f"{norm}::{src}".encode("utf-8")).hexdigest()[:32]


def build_event_record(
    title: str,
    ticker: Union[str, List[str]],
    source: str = "Google News",
    url: str = "",
    published_at: Optional[Union[str, datetime]] = None,
    first_seen_at: Optional[Union[str, datetime]] = None,
    sentiment_data: Optional[Dict[str, float]] = None,
    event_type: Optional[str] = None,
    dedup_cluster_id: Optional[str] = None,
    backfilled: bool = False,
    is_trading_day_fn: Optional[Callable[[Union[date, datetime]], bool]] = None,
    valid_trading_dates: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Constructs a validated point-in-time news event record according to the v0.7 contract.
    """
    now_utc = datetime.now(timezone.utc)
    pub_utc = _ensure_datetime_utc(published_at) if published_at else now_utc
    seen_utc = _ensure_datetime_utc(first_seen_at) if first_seen_at else pub_utc

    # available_at is the later of publication time and first detection time
    avail_utc = max(pub_utc, seen_utc)
    avail_ist = avail_utc.astimezone(IST_TZ)

    assigned_date = assign_trading_date(
        avail_ist,
        is_trading_day_fn=is_trading_day_fn,
        valid_trading_dates=valid_trading_dates,
    )

    ev_id = generate_event_id(title, source)

    if isinstance(ticker, list):
        tickers_list = ticker
        primary_ticker = ticker[0] if ticker else "UNKNOWN"
    else:
        tickers_list = [ticker]
        primary_ticker = ticker

    sent = sentiment_data or {}
    pos = float(sent.get("finbert_pos", 0.0))
    neu = float(sent.get("finbert_neu", 1.0))
    neg = float(sent.get("finbert_neg", 0.0))
    score = float(sent.get("sentiment_score", pos - neg))
    conf = float(sent.get("confidence", max(pos, neu, neg)))

    ev_type = event_type or classify_headline(title)

    return {
        "event_id": ev_id,
        "ticker": primary_ticker,
        "tickers": ",".join(tickers_list),
        "title": title.strip(),
        "source": source.strip(),
        "url": url.strip(),
        "published_at_utc": pub_utc.isoformat(),
        "first_seen_at_utc": seen_utc.isoformat(),
        "available_at": avail_ist.isoformat(),
        "trading_date_assigned": assigned_date,
        "finbert_pos": pos,
        "finbert_neu": neu,
        "finbert_neg": neg,
        "sentiment_score": score,
        "confidence": conf,
        "event_type": ev_type,
        "dedup_cluster_id": dedup_cluster_id or f"cluster_{ev_id[:12]}",
        "backfilled": bool(backfilled),
    }


class NewsEventStore:
    """
    Point-in-time Parquet-backed event store for financial news articles.
    Guarantees strict auditability, deduplication, and zero look-ahead bias.
    """

    def __init__(self, store_path: Union[str, Path] = DEFAULT_STORE_PATH):
        self.store_path = Path(store_path)

    def _ensure_dir(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def load_events(
        self,
        ticker: Optional[Union[str, List[str]]] = None,
        trading_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Loads events from Parquet store with optional filters.
        """
        if not self.store_path.exists():
            return pd.DataFrame(columns=EVENT_SCHEMA_COLUMNS)

        df = pd.read_parquet(self.store_path)
        if df.empty:
            return df

        if ticker:
            tickers = [ticker] if isinstance(ticker, str) else ticker
            df = df[df["ticker"].isin(tickers) | df["tickers"].apply(lambda t: any(x in str(t) for x in tickers))]

        if trading_date:
            df = df[df["trading_date_assigned"] == str(trading_date)]

        if start_date:
            df = df[df["trading_date_assigned"] >= str(start_date)]

        if end_date:
            df = df[df["trading_date_assigned"] <= str(end_date)]

        if event_types:
            df = df[df["event_type"].isin(event_types)]

        return df.reset_index(drop=True)

    def save_events(
        self,
        events: Union[List[Dict[str, Any]], pd.DataFrame],
        cluster_duplicates: bool = True,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Saves new events to Parquet store.
        Appends to existing records while deduplicating on 'event_id'.
        """
        self._ensure_dir()

        if isinstance(events, list):
            new_df = pd.DataFrame(events)
        else:
            new_df = events.copy()

        if new_df.empty:
            return self.load_events()

        # Ensure all columns exist
        for col in EVENT_SCHEMA_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None

        new_df = new_df[EVENT_SCHEMA_COLUMNS]

        if self.store_path.exists() and not overwrite:
            existing_df = pd.read_parquet(self.store_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by event_id, keeping the newest / updated entry
            combined_df = combined_df.drop_duplicates(subset=["event_id"], keep="last")
        else:
            combined_df = new_df.drop_duplicates(subset=["event_id"], keep="last")

        if cluster_duplicates and len(combined_df) > 1:
            records = combined_df.to_dict(orient="records")
            clustered_records = cluster_headlines(records)
            combined_df = pd.DataFrame(clustered_records)

        combined_df.to_parquet(self.store_path, index=False)
        return combined_df.reset_index(drop=True)

    def count(self) -> int:
        """Returns the total number of events in the store."""
        if not self.store_path.exists():
            return 0
        df = pd.read_parquet(self.store_path, columns=["event_id"])
        return len(df)
