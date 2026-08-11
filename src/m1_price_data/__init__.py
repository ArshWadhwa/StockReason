"""
StockReason - Module 1: Price & Market Data Pipeline (Week 1 to Week 3)
"""
from .fetcher import fetch_stock_data, fetch_multiple_stocks, fetch_index_data, fetch_macro_data
from .cleaner import clean_price_data, align_and_clean_dataset
from .technical_features import generate_technical_features
from .lag_features import generate_lag_features
from .regime_detection import detect_market_regime
from .build_features import build_pipeline

__all__ = [
    "fetch_stock_data",
    "fetch_multiple_stocks",
    "fetch_index_data",
    "fetch_macro_data",
    "clean_price_data",
    "align_and_clean_dataset",
    "generate_technical_features",
    "generate_lag_features",
    "detect_market_regime",
    "build_pipeline"
]
