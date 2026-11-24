"""
StockReason - Module 2: End-to-End Sentiment Pipeline Orchestrator
Fetches news, scores headlines with FinBERT, aggregates daily features,
computes divergence flags, and outputs production Parquet datasets.
"""
import argparse
from pathlib import Path
from typing import List, Optional
import yaml
import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    RAW_NEWS_DIR,
    CONFIG_DIR,
    TICKER_QUERY_MAP
)
from .fetcher import fetch_news_for_ticker
from .finbert_scorer import score_headlines
from .aggregator import aggregate_daily_sentiment
from .divergence import detect_sentiment_price_divergence


def load_universe_tickers() -> List[str]:
    """Loads tickers configured in config/tickers.yaml or defaults to query map."""
    tickers_file = CONFIG_DIR / "tickers.yaml"
    if tickers_file.exists():
        try:
            with open(tickers_file, "r") as f:
                cfg = yaml.safe_load(f)
                stocks = []
                if "stocks" in cfg:
                    for k, v in cfg["stocks"].items():
                        if isinstance(v, list):
                            stocks.extend(v)
                if stocks:
                    return sorted(list(set(stocks)))
        except Exception:
            pass
    return list(TICKER_QUERY_MAP.keys())


def run_sentiment_pipeline(
    tickers: Optional[List[str]] = None,
    days_back: int = 7,
    max_headlines_per_ticker: int = 20,
    use_cache: bool = True,
    use_finbert: bool = True
) -> pd.DataFrame:
    """
    Executes the full News & Sentiment pipeline.
    """
    if tickers is None:
        tickers = load_universe_tickers()
        
    print(f"[M2 Pipeline] Starting News & Sentiment pipeline for {len(tickers)} tickers...")
    
    all_scored_articles = []
    
    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] Fetching news for {ticker}...")
        raw_articles = fetch_news_for_ticker(
            ticker=ticker,
            days_back=days_back,
            max_headlines=max_headlines_per_ticker,
            use_cache=use_cache
        )
        print(f"  -> Retrieved {len(raw_articles)} articles.")
        
        if raw_articles:
            print(f"  -> Scoring sentiment (FinBERT={use_finbert})...")
            scored = score_headlines(raw_articles, use_finbert=use_finbert)
            all_scored_articles.extend(scored)

    # 1. Save all scored headlines
    if all_scored_articles:
        df_headlines = pd.DataFrame(all_scored_articles)
        if "finbert_probs" in df_headlines.columns:
            df_headlines = df_headlines.drop(columns=["finbert_probs"])
        headlines_csv_path = RAW_NEWS_DIR / "headlines.csv"
        df_headlines.to_csv(headlines_csv_path, index=False)
        print(f"[M2 Pipeline] Saved {len(df_headlines)} scored headlines to {headlines_csv_path}")
    else:
        df_headlines = pd.DataFrame()

    # 2. Aggregate into Daily Sentiment Features
    print("[M2 Pipeline] Aggregating daily sentiment features...")
    df_daily = aggregate_daily_sentiment(all_scored_articles)
    
    # 3. Check for M1 Price Data to calculate Divergence
    price_candidates = [
        PROCESSED_DATA_DIR / "price_features.csv",
        Path("data/raw/prices/pilot_stocks.csv"),
        Path("data/raw/ohlcv/pilot_stocks.csv")
    ]
    df_price = None
    for p_path in price_candidates:
        if p_path.exists():
            try:
                df_price = pd.read_csv(p_path)
                print(f"[M2 Pipeline] Found price dataset at {p_path} with {len(df_price)} records for divergence analysis.")
                break
            except Exception:
                pass
            
    if not df_daily.empty:
        df_daily = detect_sentiment_price_divergence(
            df_sentiment=df_daily,
            df_price=df_price
        )
        
    # 4. Save final sentiment_features.csv
    out_file = PROCESSED_DATA_DIR / "sentiment_features.csv"
    if not df_daily.empty:
        try:
            df_daily.to_csv(out_file, index=False)
            print(f"[M2 Pipeline] Successfully exported final daily sentiment dataset: {out_file} ({len(df_daily)} rows)")
        except Exception as exc:
            print(f"[M2 Pipeline] Failed to export CSV: {exc}")
    else:
        print("[M2 Pipeline] Warning: No sentiment features generated.")
        
    return df_daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockReason M2 News & Sentiment Pipeline")
    parser.add_argument("--fast", action="store_true", help="Use fast TextBlob baseline only without FinBERT")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local headline cache")
    args = parser.parse_args()
    
    run_sentiment_pipeline(
        use_cache=not args.no_cache,
        use_finbert=not args.fast
    )
