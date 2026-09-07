"""
StockReason - Module 2: News Fetching & Caching Pipeline
Fetches financial news headlines for stock tickers and indices using Google News RSS feeds
and optional Finnhub/NewsAPI integrations with robust local caching.
"""
import os
import json
import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import feedparser
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import (
    RAW_NEWS_DIR,
    TICKER_QUERY_MAP,
    DEFAULT_MAX_HEADLINES_PER_TICKER
)


def _clean_headline_text(text: str) -> str:
    """Removes HTML tags, extra whitespace, and trailing source attribution tags."""
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities
    import html
    cleaned = html.unescape(cleaned)
    # Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def fetch_google_news_rss(
    ticker: str,
    query: Optional[str] = None,
    max_items: int = DEFAULT_MAX_HEADLINES_PER_TICKER
) -> List[Dict[str, Any]]:
    """
    Fetches latest financial news articles for a given ticker using Google News RSS feeds.
    Free, no API key required, supports localized Indian financial media.
    """
    if not query:
        aliases = TICKER_QUERY_MAP.get(ticker, [ticker.replace(".NS", "")])
        query = aliases[0]
    
    # Target Indian business / finance context
    search_term = f"{query} stock financial market news"
    encoded_query = urllib.parse.quote(search_term)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    articles: List[Dict[str, Any]] = []
    
    for entry in feed.entries[:max_items]:
        title = _clean_headline_text(entry.get("title", ""))
        link = entry.get("link", "")
        source_info = entry.get("source", {})
        source_name = source_info.get("title", "Google News") if isinstance(source_info, dict) else "Google News"
        
        # Strip trailing source in "Headline - Source Name" pattern if present
        if " - " in title and title.rsplit(" - ", 1)[-1].strip() == source_name.strip():
            title = title.rsplit(" - ", 1)[0].strip()
        
        # Parse published date
        pub_parsed = entry.get("published_parsed")
        if pub_parsed:
            pub_dt = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
        else:
            pub_dt = datetime.now(timezone.utc)
            
        articles.append({
            "ticker": ticker,
            "title": title,
            "source": source_name,
            "url": link,
            "timestamp": pub_dt.isoformat(),
            "date": pub_dt.strftime("%Y-%m-%d"),
            "summary": _clean_headline_text(entry.get("summary", ""))
        })
        
    return articles


def fetch_finnhub_news(
    ticker: str,
    api_key: Optional[str] = None,
    fallback_key: Optional[str] = None,
    days_back: int = 7
) -> List[Dict[str, Any]]:
    """
    Fetches company news from Finnhub if an API key is available.
    Supports primary and secondary/fallback API keys if the primary key fails or hits rate limits.
    Falls back gracefully if not configured.
    """
    # Assemble available keys in priority order
    candidate_keys = []
    if api_key and api_key.strip():
        candidate_keys.append(api_key.strip())
    
    env_primary = os.getenv("FINNHUB_API_KEY", "").strip()
    if env_primary and env_primary not in candidate_keys:
        candidate_keys.append(env_primary)
        
    if fallback_key and fallback_key.strip() and fallback_key.strip() not in candidate_keys:
        candidate_keys.append(fallback_key.strip())
        
    env_fallback = (os.getenv("FINNHUB_API_KEY_FALLBACK", "") or os.getenv("FINNHUB_API_KEY_SECONDARY", "")).strip()
    if env_fallback and env_fallback not in candidate_keys:
        candidate_keys.append(env_fallback)
        
    if not candidate_keys:
        return []
    
    clean_sym = ticker.replace(".NS", "")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    for key in candidate_keys:
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_sym}&from={from_date}&to={to_date}&token={key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    articles = []
                    for item in data:
                        ts = item.get("datetime")
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
                        articles.append({
                            "ticker": ticker,
                            "title": _clean_headline_text(item.get("headline", "")),
                            "source": item.get("source", "Finnhub"),
                            "url": item.get("url", ""),
                            "timestamp": dt.isoformat(),
                            "date": dt.strftime("%Y-%m-%d"),
                            "summary": _clean_headline_text(item.get("summary", ""))
                        })
                    return articles
            # If 401/403/429 or other HTTP error, iterate to fallback key
            continue
        except Exception:
            # Network or connection failure, attempt with next key
            continue
            
    return []


def fetch_news_for_ticker(
    ticker: str,
    days_back: int = 7,
    max_headlines: int = DEFAULT_MAX_HEADLINES_PER_TICKER,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetches, deduplicates, and caches news headlines for a given ticker.
    Uses Google News RSS + Finnhub fallback.
    """
    cache_file = RAW_NEWS_DIR / f"{ticker.replace('^', 'INDEX_')}_news.json"
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                if cached_data:
                    return cached_data
        except Exception:
            pass
            
    # Fetch from Google News RSS
    articles = fetch_google_news_rss(ticker=ticker, max_items=max_headlines)
    
    # Optionally append Finnhub if available
    finnhub_articles = fetch_finnhub_news(ticker=ticker, days_back=days_back)
    articles.extend(finnhub_articles)
    
    # Deduplicate by lowercase title
    seen_titles = set()
    deduped_articles = []
    for art in articles:
        norm_title = art["title"].lower().strip()
        if norm_title and norm_title not in seen_titles:
            seen_titles.add(norm_title)
            deduped_articles.append(art)
            
    # Cache locally
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(deduped_articles, f, indent=2)
    except Exception:
        pass
        
    return deduped_articles


def fetch_all_universe_news(
    tickers: List[str],
    days_back: int = 7,
    max_per_ticker: int = DEFAULT_MAX_HEADLINES_PER_TICKER,
    use_cache: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches news for all tickers across the universe.
    Returns a dictionary mapping ticker -> list of articles.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in tickers:
        results[ticker] = fetch_news_for_ticker(
            ticker=ticker,
            days_back=days_back,
            max_headlines=max_per_ticker,
            use_cache=use_cache
        )
    return results
