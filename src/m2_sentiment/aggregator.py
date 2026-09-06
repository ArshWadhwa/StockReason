"""
StockReason - Module 2: Daily Sentiment Feature Aggregator
Aggregates article-level sentiment scores into daily stock metrics with rolling momentum,
positive/negative ratios, and decay handling.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def aggregate_daily_sentiment(
    scored_articles: List[Dict[str, Any]],
    date_range: Optional[pd.DatetimeIndex] = None,
    decay_rate: float = 0.5
) -> pd.DataFrame:
    """
    Aggregates headline sentiment into daily features per ticker.
    Matches schemas/sentiment_schema.md.
    """
    if not scored_articles:
        cols = [
            "date", "ticker", "news_count", "sentiment_mean",
            "sentiment_pos_ratio", "sentiment_neg_ratio",
            "sentiment_momentum_3d", "sentiment_momentum_7d",
            "sentiment_divergence_flag", "divergence_desc"
        ]
        return pd.DataFrame(columns=cols)

    df_raw = pd.DataFrame(scored_articles)
    
    # Ensure date column is standardized string / datetime
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
    
    # Group by ticker and date
    grouped = df_raw.groupby(["ticker", "date"])
    
    records = []
    for (ticker, date_str), group in grouped:
        count = len(group)
        mean_score = group["score_finbert"].mean()
        pos_count = (group["sentiment_label"] == "positive").sum()
        neg_count = (group["sentiment_label"] == "negative").sum()
        
        records.append({
            "ticker": ticker,
            "date": pd.to_datetime(date_str),
            "news_count": int(count),
            "sentiment_mean": float(np.round(mean_score, 4)),
            "sentiment_pos_ratio": float(np.round(pos_count / count, 4)),
            "sentiment_neg_ratio": float(np.round(neg_count / count, 4)),
        })
        
    df_daily = pd.DataFrame(records)
    if df_daily.empty:
        return df_daily

    # Sort chronologically per ticker
    df_daily = df_daily.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    # Compute rolling momentum features per ticker
    df_daily["sentiment_momentum_3d"] = (
        df_daily.groupby("ticker")["sentiment_mean"]
        .diff(3)
        .fillna(0.0)
        .round(4)
    )
    df_daily["sentiment_momentum_7d"] = (
        df_daily.groupby("ticker")["sentiment_mean"]
        .diff(7)
        .fillna(0.0)
        .round(4)
    )
    
    # Default divergence placeholders (populated by divergence.py)
    df_daily["sentiment_divergence_flag"] = 0
    df_daily["divergence_desc"] = "Neutral / No Divergence"
    
    return df_daily
