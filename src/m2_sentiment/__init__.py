"""
StockReason - Module 2: News & Sentiment Pipeline (Week 1 to Week 3)
"""
from .fetcher import (
    fetch_news_for_ticker,
    fetch_google_news_rss,
    fetch_finnhub_news,
    fetch_all_universe_news
)
from .baseline_scorer import (
    score_single_text_baseline,
    score_headlines_baseline
)
from .finbert_scorer import (
    load_finbert,
    score_texts_finbert,
    score_headlines
)
from .aggregator import aggregate_daily_sentiment
from .divergence import detect_sentiment_price_divergence
from .pipeline import run_sentiment_pipeline

__all__ = [
    "fetch_news_for_ticker",
    "fetch_google_news_rss",
    "fetch_finnhub_news",
    "fetch_all_universe_news",
    "score_single_text_baseline",
    "score_headlines_baseline",
    "load_finbert",
    "score_texts_finbert",
    "score_headlines",
    "aggregate_daily_sentiment",
    "detect_sentiment_price_divergence",
    "run_sentiment_pipeline"
]
