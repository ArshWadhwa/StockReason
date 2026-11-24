"""
StockReason v0.7 Panel Dataset Builder.

Constructs the canonical long-format panel dataset for v0.7:
- keys: ticker (str), date (datetime64, trading day, no timezone)
- price columns: open, high, low, close, adj_close, volume
- stationary feature enrichment (normalized MACD hist, relative strength, VIX dynamics)
- multi-horizon targets: y_5d, y_21d, y_5d_raw, y_21d_raw, y_5d_clipped, y_21d_clipped, etc.
- label timing columns: label_start (= t), label_end_5d, label_end_21d
"""

from typing import Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd

from .targets import compute_targets
from .feature_sets import enrich_stationary_features

logger = logging.getLogger(__name__)

UnionPath = Union[str, Path]


def build_v07_panel_dataset(
    price_parquet_path: UnionPath = "data/processed/price_features.parquet",
    sentiment_parquet_path: Optional[UnionPath] = "data/processed/sentiment_features.parquet",
    output_parquet_path: Optional[UnionPath] = "data/v0.7/panel_v07.parquet",
    horizons: Tuple[int, ...] = (5, 21),
    enable_126d: bool = False,
) -> pd.DataFrame:
    """
    Build and optionally save the v0.7 canonical panel dataset.
    """
    p_path = Path(price_parquet_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Price features parquet not found at: {p_path}")

    logger.info(f"Loading base price features from {p_path}...")
    df_price = pd.read_parquet(p_path)

    # Ensure clean timezone-naive datetime
    if "date" in df_price.columns:
        df_price["date"] = pd.to_datetime(df_price["date"]).dt.tz_localize(None)

    # Left-join sentiment features if present
    df_merged = df_price
    if sentiment_parquet_path is not None:
        s_path = Path(sentiment_parquet_path)
        if s_path.exists():
            logger.info(f"Merging sentiment features from {s_path}...")
            df_sent = pd.read_parquet(s_path)
            if "date" in df_sent.columns:
                df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.tz_localize(None)
            exclude_sent = {"ticker", "date", "close_pct_change_5d", "divergence_desc"}
            sent_cols = [c for c in df_sent.columns if c not in exclude_sent]
            df_merged = pd.merge(df_price, df_sent[["ticker", "date"] + sent_cols], on=["ticker", "date"], how="left")
            for c in sent_cols:
                df_merged[c] = df_merged[c].fillna(0)

    # Enrich stationary indicators
    logger.info("Enriching stationary technical and relative strength indicators...")
    df_enriched = enrich_stationary_features(df_merged)

    # Compute multi-horizon targets
    logger.info(f"Computing multi-horizon targets (horizons={horizons}, enable_126d={enable_126d})...")
    df_final = compute_targets(
        df=df_enriched,
        horizons=horizons,
        enable_126d=enable_126d,
    )

    # Enforce contract types
    df_final["ticker"] = df_final["ticker"].astype(str)
    df_final["date"] = pd.to_datetime(df_final["date"]).dt.tz_localize(None)

    if output_parquet_path is not None:
        out_p = Path(output_parquet_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(out_p, index=False)
        logger.info(f"Successfully saved canonical v0.7 panel dataset ({df_final.shape[0]} rows, {df_final.shape[1]} cols) -> {out_p}")

    return df_final



