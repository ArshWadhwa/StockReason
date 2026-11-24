"""
StockReason v0.7 - Module: Point-in-Time Daily News Features
Computes leak-free, point-in-time daily news and NLP sentiment features
for Indian equities per trading date.
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import pandas as pd
import numpy as np

EVENT_TYPES_RECENT = [
    "earnings",
    "regulatory",
    "guidance",
    "management_change",
    "merger_acquisition",
    "analyst_rating",
    "macro",
]

DEFAULT_FEATURE_COLUMNS = [
    "date",
    "ticker",
    "news_available",
    "news_count_1d",
    "news_count_5d",
    "news_age_days",
    "sent_mean_1d",
    "sent_decay",
    "sent_pos_ratio",
    "sent_neg_ratio",
    "event_earnings_recent",
    "event_regulatory_recent",
    "event_guidance_recent",
    "event_management_change_recent",
    "event_merger_acquisition_recent",
    "event_analyst_rating_recent",
    "event_macro_recent",
]


def compute_ticker_daily_features(
    events_df: pd.DataFrame,
    ticker: str,
    trading_dates: List[str],
    half_life_trading_days: float = 3.0,
    window_days: int = 5,
    max_age_cap: int = 30,
    format_mode: str = "tree",  # "tree" (NaNs) or "neural" (0.0 + mask)
) -> pd.DataFrame:
    """
    Computes point-in-time daily news features for a single ticker across a sequence of trading dates.

    Guarantees:
    - Features on date T depend ONLY on articles with trading_date_assigned <= T.
    - Adding articles for date T' > T will never alter features for date T.
    - Missing days yield news_available=0 and NaN sentiment (tree) or 0.0 (neural).
    """
    sorted_dates = sorted(list(set(str(d) for d in trading_dates)))

    # Filter events for this ticker
    if not events_df.empty:
        t_mask = (events_df["ticker"] == ticker) | (
            events_df["tickers"].astype(str).str.contains(ticker, regex=False)
        )
        t_events = events_df[t_mask].copy()
    else:
        t_events = pd.DataFrame()

    rows = []

    # Map trading dates to indices for decay & window calculations
    date_to_idx = {d_str: i for i, d_str in enumerate(sorted_dates)}
    decay_lambda = np.log(2.0) / float(half_life_trading_days)

    # Pre-group events by assigned trading date
    events_by_date: Dict[str, pd.DataFrame] = {}
    if not t_events.empty:
        # Filter out invalid dates
        t_events["trading_date_assigned"] = t_events["trading_date_assigned"].astype(str)
        for d_str, group in t_events.groupby("trading_date_assigned"):
            events_by_date[d_str] = group

    last_article_trading_idx: Optional[int] = None

    for i, curr_date in enumerate(sorted_dates):
        # 1. 1-day events (today)
        day_events = events_by_date.get(curr_date)
        has_day_news = day_events is not None and len(day_events) > 0

        news_count_1d = len(day_events) if has_day_news else 0
        news_available = 1 if news_count_1d > 0 else 0

        if has_day_news:
            last_article_trading_idx = i

        # 2. News Age (in trading days, capped at max_age_cap)
        if last_article_trading_idx is not None:
            news_age_days = min(float(i - last_article_trading_idx), float(max_age_cap))
        else:
            news_age_days = float(max_age_cap)

        # 3. 5-day window events
        window_start_idx = max(0, i - window_days + 1)
        window_dates = sorted_dates[window_start_idx : i + 1]

        window_event_frames = [events_by_date[d] for d in window_dates if d in events_by_date]
        if window_event_frames:
            w_events = pd.concat(window_event_frames, ignore_index=True)
            news_count_5d = len(w_events)
        else:
            w_events = pd.DataFrame()
            news_count_5d = 0

        # 4. Sentiment metrics
        if has_day_news:
            sent_mean_1d = float(day_events["sentiment_score"].mean())
        else:
            sent_mean_1d = np.nan

        if news_count_5d > 0:
            # Positive: sentiment > 0.05 or finbert_pos > finbert_neg
            pos_mask = (w_events["sentiment_score"] > 0.05) | (w_events["finbert_pos"] > w_events["finbert_neg"])
            neg_mask = (w_events["sentiment_score"] < -0.05) | (w_events["finbert_neg"] > w_events["finbert_pos"])
            sent_pos_ratio = float(pos_mask.sum()) / float(news_count_5d)
            sent_neg_ratio = float(neg_mask.sum()) / float(news_count_5d)
        else:
            sent_pos_ratio = np.nan
            sent_neg_ratio = np.nan

        # 5. Exponentially decayed sentiment over prior 30 trading days
        lookback_start_idx = max(0, i - max_age_cap + 1)
        decay_dates = sorted_dates[lookback_start_idx : i + 1]

        weighted_sent_sum = 0.0
        total_weight = 0.0

        for d_past in decay_dates:
            if d_past in events_by_date:
                past_ev = events_by_date[d_past]
                k = i - date_to_idx[d_past]  # trading days elapsed
                w = np.exp(-decay_lambda * k)
                weighted_sent_sum += w * float(past_ev["sentiment_score"].sum())
                total_weight += w * len(past_ev)

        if total_weight > 0:
            sent_decay = weighted_sent_sum / total_weight
        else:
            sent_decay = np.nan

        # 6. Recent binary event flags (over last window_days)
        recent_event_types: Set[str] = set()
        if not w_events.empty and "event_type" in w_events.columns:
            recent_event_types = set(w_events["event_type"].dropna().unique())

        event_flags = {
            f"event_{ev}_recent": 1 if ev in recent_event_types else 0
            for ev in EVENT_TYPES_RECENT
        }

        # 7. Format handling: tree (NaNs) vs neural (0.0 fill)
        if format_mode == "neural":
            sent_mean_1d = 0.0 if np.isnan(sent_mean_1d) else sent_mean_1d
            sent_decay = 0.0 if np.isnan(sent_decay) else sent_decay
            sent_pos_ratio = 0.0 if np.isnan(sent_pos_ratio) else sent_pos_ratio
            sent_neg_ratio = 0.0 if np.isnan(sent_neg_ratio) else sent_neg_ratio

        row = {
            "date": curr_date,
            "ticker": ticker,
            "news_available": news_available,
            "news_count_1d": news_count_1d,
            "news_count_5d": news_count_5d,
            "news_age_days": news_age_days,
            "sent_mean_1d": sent_mean_1d,
            "sent_decay": sent_decay,
            "sent_pos_ratio": sent_pos_ratio,
            "sent_neg_ratio": sent_neg_ratio,
            **event_flags,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_daily_news_features(
    events_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    trading_dates: Optional[List[str]] = None,
    half_life_trading_days: float = 3.0,
    window_days: int = 5,
    max_age_cap: int = 30,
    format_mode: str = "tree",
) -> pd.DataFrame:
    """
    Builds the complete daily point-in-time news features dataset across tickers and dates.
    """
    if tickers is None:
        if not events_df.empty:
            tickers = sorted(events_df["ticker"].dropna().unique().tolist())
        else:
            tickers = []

    if trading_dates is None:
        if not events_df.empty:
            trading_dates = sorted(events_df["trading_date_assigned"].dropna().unique().tolist())
        else:
            trading_dates = []

    all_dfs = []
    for ticker in tickers:
        df_ticker = compute_ticker_daily_features(
            events_df=events_df,
            ticker=ticker,
            trading_dates=trading_dates,
            half_life_trading_days=half_life_trading_days,
            window_days=window_days,
            max_age_cap=max_age_cap,
            format_mode=format_mode,
        )
        all_dfs.append(df_ticker)

    if all_dfs:
        res = pd.concat(all_dfs, ignore_index=True)
        res["date"] = pd.to_datetime(res["date"])
        return res.sort_values(["date", "ticker"]).reset_index(drop=True)
    return pd.DataFrame(columns=DEFAULT_FEATURE_COLUMNS)
