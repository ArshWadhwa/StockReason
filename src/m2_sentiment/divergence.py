"""
StockReason - Module 2: Sentiment-Price Divergence Engine
Differentiator add-on that identifies market mispricings where price action
diverges significantly from underlying financial sentiment.
"""
from typing import Optional
import pandas as pd
import numpy as np


def detect_sentiment_price_divergence(
    df_sentiment: pd.DataFrame,
    df_price: Optional[pd.DataFrame] = None,
    price_return_col: str = "close_pct_change_5d",
    sentiment_col: str = "sentiment_mean",
    price_dip_thresh: float = -0.02,     # -2% price dip
    price_rally_thresh: float = 0.02,    # +2% price rally
    sent_pos_thresh: float = 0.15,       # Positive sentiment threshold
    sent_neg_thresh: float = -0.15       # Negative sentiment threshold
) -> pd.DataFrame:
    """
    Computes Sentiment-Price Divergence flags:
      +1 : Bullish Divergence (Price falling while news sentiment is bullish/improving)
      -1 : Bearish Divergence (Price rising while news sentiment is bearish/deteriorating)
       0 : Normal / Aligned (No divergence)
    """
    df = df_sentiment.copy()
    if df.empty:
        return df

    # If price DataFrame is provided, merge on [ticker, date]
    if df_price is not None and not df_price.empty:
        df_p = df_price.copy()
        df_p["date"] = pd.to_datetime(df_p["date"])
        df["date"] = pd.to_datetime(df["date"])
        
        # Calculate 5-day price return if not precomputed
        if price_return_col not in df_p.columns and "close" in df_p.columns:
            df_p = df_p.sort_values(by=["ticker", "date"])
            df_p[price_return_col] = df_p.groupby("ticker")["close"].pct_change(5)
            
        merge_cols = ["ticker", "date"]
        if price_return_col in df_p.columns:
            merge_cols.append(price_return_col)
            
        df = pd.merge(df, df_p[merge_cols], on=["ticker", "date"], how="left")
    
    # Check if price return column exists
    if price_return_col not in df.columns:
        # If no price data provided, return neutral divergence flags
        df["sentiment_divergence_flag"] = 0
        df["divergence_desc"] = "Price data unavailable for divergence calculation"
        return df

    # Calculate divergence flags
    conditions = [
        # Bullish Divergence
        (df[price_return_col] <= price_dip_thresh) & (df[sentiment_col] >= sent_pos_thresh),
        # Bearish Divergence
        (df[price_return_col] >= price_rally_thresh) & (df[sentiment_col] <= sent_neg_thresh)
    ]
    
    choices = [1, -1]
    df["sentiment_divergence_flag"] = np.select(conditions, choices, default=0)
    
    # Generate human-readable explanation
    def explain_row(row):
        flag = row.get("sentiment_divergence_flag", 0)
        p_ret = row.get(price_return_col, 0.0)
        s_score = row.get(sentiment_col, 0.0)
        p_str = f"{p_ret * 100:.1f}%" if pd.notnull(p_ret) else "N/A"
        
        if flag == 1:
            return f"Bullish Divergence: Price dropped ({p_str}) despite positive sentiment ({s_score:.2f})"
        elif flag == -1:
            return f"Bearish Divergence: Price gained ({p_str}) despite negative sentiment ({s_score:.2f})"
        return "Normal: Price and sentiment are in alignment"
        
    df["divergence_desc"] = df.apply(explain_row, axis=1)
    
    return df
