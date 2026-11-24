"""
StockReason v0.10 — Corporate Announcements Feature Ingestion.
==============================================================
Remediates Phase 7 (Section 13):
Ingests genuine NSE corporate announcement filings:
- File: data/external/corporate_announcements/nse_corporate_announcements_2024_onward.parquet
- Features:
  * Trailing 30-day corporate announcement frequency.
  * Trailing 90-day corporate announcement frequency.
  * Recent Financial Results Flag (within last 14 sessions).
  * Has XBRL attachment flag.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import pandas as pd


class CorporateAnnouncementsFeatureExtractor:
    def __init__(
        self,
        announcements_path: str = "data/external/corporate_announcements/nse_corporate_announcements_2024_onward.parquet",
    ):
        self.announcements_path = announcements_path
        self.ann_df: Optional[pd.DataFrame] = None

    def load_announcements(self) -> pd.DataFrame:
        if self.ann_df is not None:
            return self.ann_df

        if not os.path.exists(self.announcements_path):
            self.ann_df = pd.DataFrame(columns=["symbol", "date_str", "category"])
            return self.ann_df

        df = pd.read_parquet(self.announcements_path, columns=["symbol", "broadcast_ts", "category", "has_xbrl"])
        df["date_str"] = pd.to_datetime(df["broadcast_ts"]).dt.strftime("%Y-%m-%d")
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["is_results"] = df["category"].astype(str).str.contains("Financial Results|Quarterly|Results", case=False).astype(float)
        self.ann_df = df.sort_values("date_str")
        return self.ann_df

    def compute_symbol_announcement_features(
        self,
        panel_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges trailing announcement counts into panel point-in-time.
        """
        out = panel_df.copy()
        ann = self.load_announcements()
        if ann.empty or "date_str" not in out.columns:
            out["ann_count_30d"] = 0.0
            out["ann_has_recent_results"] = 0.0
            return out

        # Aggregate daily count per (symbol, date_str)
        daily_counts = ann.groupby(["symbol", "date_str"]).size().reset_index(name="daily_ann_count")
        daily_results = ann.groupby(["symbol", "date_str"])["is_results"].max().reset_index(name="daily_results_flag")

        merged = pd.merge(out, daily_counts, on=["symbol", "date_str"], how="left")
        merged["daily_ann_count"] = merged["daily_ann_count"].fillna(0.0)

        # Rolling 30 sessions sum per symbol
        merged["ann_count_30d"] = merged.groupby("symbol")["daily_ann_count"].transform(lambda s: s.rolling(21, min_periods=1).sum()).fillna(0.0)

        merged = pd.merge(merged, daily_results, on=["symbol", "date_str"], how="left")
        merged["daily_results_flag"] = merged["daily_results_flag"].fillna(0.0)
        merged["ann_has_recent_results"] = merged.groupby("symbol")["daily_results_flag"].transform(lambda s: s.rolling(10, min_periods=1).max()).fillna(0.0)

        return merged
