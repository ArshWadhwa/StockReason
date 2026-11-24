"""
StockReason v0.7 - News & NLP Package
Point-in-Time Event Store, Rule-Based Event Typing, Syndication Deduplication,
and Leak-Free Daily Sentiment Feature Engineering.
"""

from .store import (
    NewsEventStore,
    assign_trading_date,
    build_event_record,
    generate_event_id,
    EVENT_SCHEMA_COLUMNS,
    DEFAULT_STORE_PATH,
)
from .classifier import (
    classify_headline,
    classify_batch,
    get_classification_details,
    EVENT_TYPES,
)
from .dedup import (
    normalize_title,
    title_similarity,
    cluster_headlines,
    deduplicate_dataframe,
)
from .sources import (
    GoogleNewsRSSSource,
    GDELTSource,
    LocalRawNewsAdapter,
    ingest_articles_to_store,
)
from .features import (
    build_daily_news_features,
    compute_ticker_daily_features,
    DEFAULT_FEATURE_COLUMNS,
)

__all__ = [
    "NewsEventStore",
    "assign_trading_date",
    "build_event_record",
    "generate_event_id",
    "EVENT_SCHEMA_COLUMNS",
    "DEFAULT_STORE_PATH",
    "classify_headline",
    "classify_batch",
    "get_classification_details",
    "EVENT_TYPES",
    "normalize_title",
    "title_similarity",
    "cluster_headlines",
    "deduplicate_dataframe",
    "GoogleNewsRSSSource",
    "GDELTSource",
    "LocalRawNewsAdapter",
    "ingest_articles_to_store",
    "build_daily_news_features",
    "compute_ticker_daily_features",
    "DEFAULT_FEATURE_COLUMNS",
]
