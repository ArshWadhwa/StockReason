"""
Builds and saves the universe metadata table: data/v0.7/metadata/universe.parquet.
Contains ticker, company, sector, tier, date range, row counts, median traded value,
exclusion reason, and survivorship bias declarations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.v07.data_expansion.universe import (
    get_universe_details,
    SURVIVORSHIP_SAFE,
    SURVIVORSHIP_BIAS_NOTE,
)
from src.v07.data_expansion.cleaner import calculate_median_traded_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ws6.metadata")

DEFAULT_METADATA_PATH = "data/v0.7/metadata/universe.parquet"


def build_universe_metadata_table(
    panel_df: Optional[pd.DataFrame] = None,
    excluded_info: Optional[Dict[str, Dict[str, Any]]] = None,
    tier: str = "U300",
    output_path: str = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """
    Builds the standardized universe metadata table combining constituent definitions,
    historical coverage statistics, liquidity metrics, and exclusion reasons.

    Schema:
    - ticker: str
    - company: str
    - sector: str
    - tier: str (highest membership tier: U16, U100, U200, U300)
    - first_date: str (YYYY-MM-DD)
    - last_date: str (YYYY-MM-DD)
    - rows: int
    - median_traded_value_inr: float
    - median_traded_value_cr: float
    - is_included: bool
    - exclusion_reason: Optional[str]
    - survivorship_safe: bool (False)
    """
    constituents_df = get_universe_details(tier).copy()
    excluded = excluded_info or {}

    # Extract coverage stats from panel if provided
    coverage: Dict[str, Dict[str, Any]] = {}
    medians: Dict[str, float] = {}

    if panel_df is not None and not panel_df.empty and "ticker" in panel_df.columns:
        medians = calculate_median_traded_value(panel_df, window=60)
        for t, g in panel_df.groupby("ticker"):
            coverage[t] = {
                "rows": len(g),
                "first_date": str(g["date"].min().date()) if pd.notnull(g["date"].min()) else None,
                "last_date": str(g["date"].max().date()) if pd.notnull(g["date"].max()) else None,
            }

    records = []
    for _, row in constituents_df.iterrows():
        t = row["ticker"]
        comp = row["company"]
        sec = row["sector"]
        tr = row["tier"]

        cov = coverage.get(t, {})
        ex = excluded.get(t, {})

        is_ex = t in excluded
        rows_count = cov.get("rows", ex.get("rows", 0))
        first_d = cov.get("first_date", ex.get("first_date", None))
        last_d = cov.get("last_date", ex.get("last_date", None))

        med_val = medians.get(t, ex.get("median_traded_value_inr", 0.0))
        med_cr = med_val / 10_000_000.0 if med_val > 0 else 0.0

        reason = ex.get("reason", None) if is_ex else None
        is_inc = not is_ex and (rows_count > 0)

        if not is_inc and reason is None:
            reason = "not_downloaded_or_empty"

        records.append({
            "ticker": t,
            "company": comp,
            "sector": sec,
            "tier": tr,
            "first_date": first_d,
            "last_date": last_d,
            "rows": int(rows_count),
            "median_traded_value_inr": float(med_val),
            "median_traded_value_cr": float(round(med_cr, 4)),
            "is_included": bool(is_inc),
            "exclusion_reason": reason,
            "survivorship_safe": SURVIVORSHIP_SAFE,
        })

    meta_df = pd.DataFrame(records)
    meta_df.attrs["survivorship_safe"] = SURVIVORSHIP_SAFE
    meta_df.attrs["survivorship_bias_note"] = SURVIVORSHIP_BIAS_NOTE

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta_df.to_parquet(output_path, index=False)
    logger.info(f"Saved universe metadata table ({len(meta_df)} constituents) to {output_path}")

    # Also save human-readable CSV for easy inspection alongside parquet
    csv_path = output_path.replace(".parquet", ".csv")
    meta_df.to_csv(csv_path, index=False)

    return meta_df
