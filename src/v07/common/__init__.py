"""
StockReason v0.7 Common Package
Owned by WS1 (Data Integrity Engineer).
"""

from src.v07.common.config import get_project_root, load_v07_config
from src.v07.common.freshness import (
    FreshnessReport,
    SourceFreshness,
    StaleDataError,
    check_freshness,
    enforce_freshness_for_forecast,
    get_api_freshness_payload,
)
from src.v07.common.sequences import (
    SequenceDataset,
    build_sequences,
    get_live_inference_window,
)
from src.v07.common.validation import (
    ValidationReport,
    count_missed_trading_days_between,
    count_trading_days_elapsed,
    validate_dataset_integrity,
    validate_price_moves,
    validate_trading_gaps,
)

__all__ = [
    "get_project_root",
    "load_v07_config",
    "FreshnessReport",
    "SourceFreshness",
    "StaleDataError",
    "check_freshness",
    "enforce_freshness_for_forecast",
    "get_api_freshness_payload",
    "SequenceDataset",
    "build_sequences",
    "get_live_inference_window",
    "ValidationReport",
    "count_missed_trading_days_between",
    "count_trading_days_elapsed",
    "validate_dataset_integrity",
    "validate_price_moves",
    "validate_trading_gaps",
]
