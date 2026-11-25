"""StockReason Feature Engineering Package."""

from src.features.registry import DETAILED_FEATURES
from src.features.technical import compute_technical_features
from src.features.market import merge_market_and_sector_features, load_market_benchmark_indices
from src.features.fundamental import merge_point_in_time_financials, load_and_clean_annual_fundamentals
from src.features.ownership import merge_point_in_time_ownership, load_clean_shareholding_data
from src.features.liquidity import compute_liquidity_features
from src.features.pipeline import assemble_canonical_features, POPULATED_FEATURE_NAMES

__all__ = [
    "DETAILED_FEATURES",
    "POPULATED_FEATURE_NAMES",
    "compute_technical_features",
    "merge_market_and_sector_features",
    "load_market_benchmark_indices",
    "merge_point_in_time_financials",
    "load_and_clean_annual_fundamentals",
    "merge_point_in_time_ownership",
    "load_clean_shareholding_data",
    "compute_liquidity_features",
    "assemble_canonical_features",
]
