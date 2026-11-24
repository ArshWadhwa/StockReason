"""
StockReason v0.7 - Module: News Sources and Historical Adapters
Connectors for free news data sources:
- Google News RSS feed adapter
- GDELT 2.0 Doc API adapter
- Local raw news adapter (for offline backfill from data/raw/news/)
Handles rate limits, network timeouts, and malformed responses gracefully.
"""
import os
import json
import logging
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import feedparser

from src.m2_sentiment.config import TICKER_QUERY_MAP
from src.m2_sentiment.finbert_scorer import score_texts_finbert
from src.v07.news.store import build_event_record, NewsEventStore
from src.v07.news.classifier import classify_headline

logger = logging.getLogger("v07_news_sources")


class GoogleNewsRSSSource:
    """
    Free adapter for Google News RSS feeds.
    Localizes queries to Indian financial markets (hl=en-IN&gl=IN&ceid=IN:en).
    """

    def __init__(self, timeout_sec: int = 10):
        self.timeout = timeout_sec

    def fetch(
        self,
        ticker: str,
        query: Optional[str] = None,
        max_items: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Fetches headlines from Google News RSS for a ticker.
        Returns a list of raw article dictionaries.
        """
        if not query:
            aliases = TICKER_QUERY_MAP.get(ticker, [ticker.replace(".NS", "")])
            query = aliases[0]

        search_query = f"{query} stock financial market news"
        encoded_query = urllib.parse.quote(search_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            # feedparser can parse url directly or via fetched text with timeout
            resp = requests.get(rss_url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"Google News RSS returned status {resp.status_code} for {ticker}")
                return []

            feed = feedparser.parse(resp.content)
            articles = []

            for entry in feed.entries[:max_items]:
                raw_title = entry.get("title", "")
                link = entry.get("link", "")
                source_dict = entry.get("source", {})
                source_name = source_dict.get("title", "Google News") if isinstance(source_dict, dict) else "Google News"

                pub_parsed = entry.get("published_parsed")
                if pub_parsed:
                    pub_dt = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = datetime.now(timezone.utc)

                articles.append({
                    "title": raw_title,
                    "url": link,
                    "source": source_name,
                    "published_at_utc": pub_dt.isoformat(),
                    "first_seen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                })

            return articles

        except Exception as e:
            logger.warning(f"Google News RSS fetch failed for {ticker}: {e}")
            return []


class GDELTSource:
    """
    Adapter for GDELT 2.0 Doc API (Global Database of Events, Language, and Tone).
    Free, no API key required, public financial media query endpoint.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, timeout_sec: int = 10):
        self.timeout = timeout_sec

    def fetch(
        self,
        ticker: str,
        query: Optional[str] = None,
        max_records: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Fetches global/Indian financial articles from GDELT 2.0 API.
        """
        if not query:
            aliases = TICKER_QUERY_MAP.get(ticker, [ticker.replace(".NS", "")])
            query = aliases[0]

        # Target English articles mentioning company and financial terms
        full_query = f'"{query}" (earnings OR profit OR revenue OR shares OR stock) sourcelang:english'
        params = {
            "query": full_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(max_records, 50),
            "sort": "DateDesc",
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                logger.warning(f"GDELT rate limit encountered for {ticker} (HTTP 429).")
                return []
            if resp.status_code != 200:
                logger.warning(f"GDELT API returned HTTP {resp.status_code} for {ticker}")
                return []

            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                title = item.get("title", "")
                url = item.get("url", "")
                domain = item.get("domain", "GDELT")
                seendate = item.get("seendate", "")

                # GDELT date format: YYYYMMDDTHHMMSSZ
                try:
                    pub_dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                articles.append({
                    "title": title,
                    "url": url,
                    "source": domain,
                    "published_at_utc": pub_dt.isoformat(),
                    "first_seen_at_utc": pub_dt.isoformat(),
                    "ticker": ticker,
                })

            return articles

        except Exception as e:
            logger.warning(f"GDELT fetch failed for {ticker}: {e}")
            return []


class LocalRawNewsAdapter:
    """
    Adapter that parses existing raw news files from data/raw/news/
    and converts them into standard point-in-time event records for the store.
    """

    def __init__(self, raw_dir: str = "data/raw/news"):
        self.raw_dir = Path(raw_dir)

    def load_raw_articles(self) -> List[Dict[str, Any]]:
        """
        Loads all headlines from headlines.parquet or {ticker}_news.json files.
        """
        articles = []
        parquet_path = self.raw_dir / "headlines.parquet"
        if parquet_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                for _, row in df.iterrows():
                    articles.append({
                        "ticker": row.get("ticker", "UNKNOWN"),
                        "title": row.get("title", ""),
                        "source": row.get("source", "Google News"),
                        "url": row.get("url", ""),
                        "published_at_utc": row.get("timestamp", row.get("date", "")),
                        "score_finbert": row.get("score_finbert", 0.0),
                        "confidence": row.get("confidence", 0.5),
                        "sentiment_label": row.get("sentiment_label", "neutral"),
                    })
                return articles
            except Exception as e:
                logger.warning(f"Failed to read raw headlines.parquet: {e}")

        # Fallback to json files
        for json_file in self.raw_dir.glob("*_news.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    ticker = data.get("ticker", json_file.stem.replace("_news", ""))
                    for item in data.get("articles", []):
                        articles.append({
                            "ticker": ticker,
                            "title": item.get("title", ""),
                            "source": item.get("source", "Google News"),
                            "url": item.get("url", ""),
                            "published_at_utc": item.get("published_at_utc", item.get("date", "")),
                            "score_finbert": item.get("score_finbert", 0.0),
                            "confidence": item.get("confidence", 0.5),
                        })
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")

        return articles


def ingest_articles_to_store(
    articles: List[Dict[str, Any]],
    store: Optional[NewsEventStore] = None,
    backfilled: bool = True,
    score_with_finbert: bool = True,
) -> int:
    """
    Scores raw articles, maps them to the v0.7 point-in-time schema, and persists to store.
    Returns count of saved events.
    """
    if not articles:
        return 0

    store = store or NewsEventStore()

    # Score headlines if needed
    titles = [a["title"] for a in articles]
    need_scoring = score_with_finbert and any("score_finbert" not in a for a in articles)

    scores = []
    if need_scoring:
        try:
            scores = score_texts_finbert(titles)
        except Exception as e:
            logger.warning(f"FinBERT scoring failed in ingestion: {e}")
            scores = []

    events = []
    for idx, art in enumerate(articles):
        title = art.get("title", "")
        if not title:
            continue

        sent_info = {}
        if idx < len(scores) and scores:
            sc = scores[idx]
            sent_info = {
                "finbert_pos": sc.get("pos", 0.0),
                "finbert_neu": sc.get("neu", 1.0),
                "finbert_neg": sc.get("neg", 0.0),
                "sentiment_score": sc.get("score", 0.0),
                "confidence": sc.get("confidence", 0.5),
            }
        else:
            # Fallback to scores inside art
            finbert_score = float(art.get("score_finbert", 0.0))
            conf = float(art.get("confidence", 0.5))
            pos = max(0.0, finbert_score)
            neg = max(0.0, -finbert_score)
            neu = max(0.0, 1.0 - (pos + neg))
            sent_info = {
                "finbert_pos": pos,
                "finbert_neu": neu,
                "finbert_neg": neg,
                "sentiment_score": finbert_score,
                "confidence": conf,
            }

        rec = build_event_record(
            title=title,
            ticker=art.get("ticker", "UNKNOWN"),
            source=art.get("source", "Google News"),
            url=art.get("url", ""),
            published_at=art.get("published_at_utc"),
            first_seen_at=art.get("first_seen_at_utc"),
            sentiment_data=sent_info,
            event_type=classify_headline(title),
            backfilled=backfilled,
        )
        events.append(rec)

    saved_df = store.save_events(events)
    return len(saved_df)
