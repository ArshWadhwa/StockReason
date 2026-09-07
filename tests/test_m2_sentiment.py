"""
Unit tests for StockReason Module 2 (News & Sentiment Track)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.m2_sentiment.baseline_scorer import score_single_text_baseline, score_headlines_baseline
from src.m2_sentiment.finbert_scorer import score_texts_finbert
from src.m2_sentiment.aggregator import aggregate_daily_sentiment
from src.m2_sentiment.divergence import detect_sentiment_price_divergence
from src.m2_sentiment.fetcher import fetch_finnhub_news
from unittest.mock import patch, MagicMock


# Sample financial headline fixtures
SAMPLE_HEADLINES = [
    {
        "ticker": "RELIANCE.NS",
        "title": "Reliance Q3 net profit surges 12% on strong retail and telecom earnings",
        "source": "Economic Times",
        "url": "https://example.com/1",
        "timestamp": "2024-01-15T09:30:00+00:00",
        "date": "2024-01-15"
    },
    {
        "ticker": "RELIANCE.NS",
        "title": "Reliance Industries expands green hydrogen capacity with new investment",
        "source": "Livemint",
        "url": "https://example.com/2",
        "timestamp": "2024-01-15T14:15:00+00:00",
        "date": "2024-01-15"
    },
    {
        "ticker": "INFY.NS",
        "title": "Infosys cuts annual revenue guidance as client discretionary spending slows",
        "source": "Reuters",
        "url": "https://example.com/3",
        "timestamp": "2024-01-15T10:00:00+00:00",
        "date": "2024-01-15"
    }
]


def test_baseline_scorer_positive_and_negative():
    pos_res = score_single_text_baseline("Company reports stellar earnings and massive record profits")
    assert pos_res["score_textblob"] > 0
    assert pos_res["sentiment_label"] == "positive"
    
    neg_res = score_single_text_baseline("Company suffers severe losses and major product recalls")
    assert neg_res["score_textblob"] < 0
    assert neg_res["sentiment_label"] == "negative"


def test_baseline_headlines_batch():
    scored = score_headlines_baseline(SAMPLE_HEADLINES)
    assert len(scored) == len(SAMPLE_HEADLINES)
    for item in scored:
        assert "score_textblob" in item
        assert "sentiment_label" in item
        assert "confidence" in item
        assert -1.0 <= item["score_textblob"] <= 1.0


def test_finbert_financial_semantics():
    texts = [
        "Company quarterly profit jumps 25% beating all analyst estimates",
        "Company faces huge regulatory penalty and lawsuit amid accounting fraud",
        "The board meeting will be held on Thursday afternoon"
    ]
    results = score_texts_finbert(texts)
    assert len(results) == 3
    # First should be positive
    assert results[0]["sentiment_label"] == "positive"
    assert results[0]["score_finbert"] > 0
    # Second should be negative
    assert results[1]["sentiment_label"] == "negative"
    assert results[1]["score_finbert"] < 0
    # Third should be neutral
    assert results[2]["sentiment_label"] == "neutral"


def test_aggregator_daily_metrics():
    scored = score_headlines_baseline(SAMPLE_HEADLINES)
    df_daily = aggregate_daily_sentiment(scored)
    
    assert not df_daily.empty
    assert "sentiment_mean" in df_daily.columns
    assert "news_count" in df_daily.columns
    assert "sentiment_pos_ratio" in df_daily.columns
    assert "sentiment_neg_ratio" in df_daily.columns
    assert "sentiment_momentum_3d" in df_daily.columns
    
    # Check Reliance aggregation (2 articles)
    rel_rows = df_daily[df_daily["ticker"] == "RELIANCE.NS"]
    assert len(rel_rows) == 1
    assert rel_rows.iloc[0]["news_count"] == 2


def test_sentiment_price_divergence():
    # Setup mock sentiment and price
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df_sent = pd.DataFrame({
        "ticker": ["RELIANCE.NS"] * 6,
        "date": dates,
        "sentiment_mean": [0.60, 0.70, 0.65, 0.80, 0.75, 0.85]  # Strongly positive news
    })
    
    # Price plummets from 100 to 90 (-10%)
    df_price = pd.DataFrame({
        "ticker": ["RELIANCE.NS"] * 6,
        "date": dates,
        "close": [100.0, 98.0, 95.0, 93.0, 91.0, 90.0]
    })
    
    df_div = detect_sentiment_price_divergence(
        df_sentiment=df_sent,
        df_price=df_price,
        price_dip_thresh=-0.02,
        sent_pos_thresh=0.20
    )
    
    assert "sentiment_divergence_flag" in df_div.columns
    # Row 5 (after 5 days of drop) should flag Bullish Divergence (+1)
    last_row = df_div.iloc[-1]
    assert last_row["sentiment_divergence_flag"] == 1
    assert "Bullish Divergence" in last_row["divergence_desc"]


def test_finnhub_key_fallback():
    # Primary key returns 429 Rate Limit, secondary key returns 200 with news
    def mock_requests_get(url, **kwargs):
        mock_resp = MagicMock()
        if "token=primary_invalid" in url:
            mock_resp.status_code = 429
        elif "token=secondary_valid" in url:
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {
                    "headline": "Fallback news article arrived safely",
                    "source": "Finnhub",
                    "url": "https://finnhub.io/news/1",
                    "datetime": 1700000000,
                    "summary": "Everything is normal."
                }
            ]
        else:
            mock_resp.status_code = 401
        return mock_resp

    with patch("requests.get", side_effect=mock_requests_get):
        articles = fetch_finnhub_news(
            ticker="RELIANCE.NS",
            api_key="primary_invalid",
            fallback_key="secondary_valid"
        )
        assert len(articles) == 1
        assert articles[0]["title"] == "Fallback news article arrived safely"
        assert articles[0]["source"] == "Finnhub"
