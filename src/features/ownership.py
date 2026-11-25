"""StockReason v0.13B — Ownership, Delivery & Market-Structure Intelligence.

Implements Section 9 Requirements:
1. Ownership Intelligence (from BSE XBRL Filings):
   - Promoter ownership level and QoQ/YoY changes.
   - Institutional ownership changes (FII, DII, combined institutional).
   - Public shareholding level and changes.
   - Point-in-time publication timestamp from XBRL filing metadata.
   - Feature age and staleness tracking (age > 120 days).
2. Delivery Intelligence (from NSE Daily Settlement Archives):
   - Delivery percentage (delivery_qty / volume).
   - 20-day delivery moving average and abnormal delivery ratio.
   - Delivery percentage spread relative to trailing baseline.
3. Market Structure & Liquidity:
   - Turnover ratio (20-day moving average relative turnover).
   - Amihud illiquidity proxy: |ret_1d| / turnover.
   - Trading suspension and resumption indicators based on calendar session gaps.
4. Governance Safeguards:
   - Zero lookahead: shareholding is available strictly after filing publication date.
   - Delivery is treated as structural hypothesis, not confirmed institutional buying.
"""

from __future__ import annotations

import glob
import gc
import logging
import os
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013b_ownership_delivery")

FN_DATE_REGEX = re.compile(r'_(\d{2})(\d{2})(\d{4})\d+_WEB\.xml')


def parse_shareholding_publication_date(row: pd.Series) -> pd.Timestamp:
    """Extracts verified publication date from submission_date or XBRL filename."""
    sub = row.get("submission_date")
    if pd.notna(sub) and str(sub).strip():
        try:
            return pd.to_datetime(sub, format="%d-%b-%Y")
        except Exception:
            pass

    fn = str(row.get("source_filename", ""))
    m = FN_DATE_REGEX.search(fn)
    if m:
        d, m_str, y = m.group(1), m.group(2), m.group(3)
        return pd.to_datetime(f"{y}-{m_str}-{d}")

    # Fallback to period + 21 days (SEBI Reg 31 LODR filing deadline)
    p = str(row.get("period", ""))
    try:
        return pd.to_datetime(p + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=21)
    except Exception:
        return pd.NaT


def load_clean_shareholding_data(
    flat_csv_path: str = "data/_flat.csv",
) -> pd.DataFrame:
    """Loads and cleans XBRL shareholding patterns with exact publication timestamps."""
    if not os.path.exists(flat_csv_path):
        logger.warning("Shareholding file %s not found", flat_csv_path)
        return pd.DataFrame()

    df = pd.read_csv(flat_csv_path)
    df["symbol"] = df["ticker"].astype(str)
    df["publication_date"] = df.apply(parse_shareholding_publication_date, axis=1)
    df = df.dropna(subset=["symbol", "publication_date"]).sort_values(["symbol", "publication_date"]).reset_index(drop=True)

    # Ownership percentages
    df["promoter_pct"] = pd.to_numeric(df["promoter_pct"], errors="coerce")
    df["fii_pct"] = pd.to_numeric(df["fii_pct"], errors="coerce").fillna(0.0)
    df["dii_pct"] = pd.to_numeric(df["dii_pct"], errors="coerce").fillna(0.0)
    df["public_pct"] = pd.to_numeric(df["public_pct"], errors="coerce")
    df["institutional_pct"] = df["fii_pct"] + df["dii_pct"]

    # QoQ and YoY deltas
    df["promoter_lag1"] = df.groupby("symbol")["promoter_pct"].shift(1)
    df["promoter_lag4"] = df.groupby("symbol")["promoter_pct"].shift(4)
    df["promoter_change_qoq"] = df["promoter_pct"] - df["promoter_lag1"]
    df["promoter_change_yoy"] = df["promoter_pct"] - df["promoter_lag4"]

    df["fii_lag1"] = df.groupby("symbol")["fii_pct"].shift(1)
    df["fii_change_qoq"] = df["fii_pct"] - df["fii_lag1"]

    df["dii_lag1"] = df.groupby("symbol")["dii_pct"].shift(1)
    df["dii_change_qoq"] = df["dii_pct"] - df["dii_lag1"]

    df["inst_lag1"] = df.groupby("symbol")["institutional_pct"].shift(1)
    df["institutional_change_qoq"] = df["institutional_pct"] - df["inst_lag1"]

    df["public_lag1"] = df.groupby("symbol")["public_pct"].shift(1)
    df["public_change_qoq"] = df["public_pct"] - df["public_lag1"]

    df["smart_money_delta"] = (df["fii_change_qoq"] + df["dii_change_qoq"]) - df["public_change_qoq"].fillna(0.0)

    keep_cols = [
        "symbol", "publication_date", "period",
        "promoter_pct", "promoter_change_qoq", "promoter_change_yoy",
        "fii_pct", "fii_change_qoq", "dii_pct", "dii_change_qoq",
        "institutional_pct", "institutional_change_qoq",
        "public_pct", "public_change_qoq", "smart_money_delta",
    ]
    clean_sh = df[keep_cols].copy()
    logger.info("Prepared clean shareholding data: %d filings across %d symbols", len(clean_sh), clean_sh["symbol"].nunique())
    return clean_sh


def load_delivery_features_for_panel(
    panel_df: pd.DataFrame,
    delivery_dir: str = "data/indian-markets/delivery/nse",
) -> pd.DataFrame:
    """
    Merges historical daily delivery statistics onto the panel and calculates
    abnormal delivery activity, liquidity, and trading suspension/resumption indicators.
    """
    logger.info("Calculating market structure & delivery features for panel (%d rows)...", len(panel_df))
    panel = panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    panel["symbol"] = panel["symbol"].astype(str)

    # Load delivery data efficiently
    deliv_files = sorted(glob.glob(os.path.join(delivery_dir, "*/*.parquet")))
    if not deliv_files:
        logger.warning("No delivery files found in %s", delivery_dir)
        panel["delivery_pct"] = np.nan
        panel["abnormal_delivery_ratio"] = np.nan
        panel["is_trading_resumption"] = False
        panel["is_prolonged_suspension"] = False
        return panel

    panel_syms = set(panel["symbol"].unique())
    dfs = []
    for f in deliv_files:
        d = pd.read_parquet(f, columns=["date", "symbol", "delivery_qty", "delivery_pct"])
        d = d[d["symbol"].isin(panel_syms)]
        if len(d) > 0:
            dfs.append(d)
    del_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    del_all["date"] = pd.to_datetime(del_all["date"]).astype("datetime64[ns]")
    del_all["symbol"] = del_all["symbol"].astype(str)
    del_all["delivery_qty"] = del_all["delivery_qty"].astype("float32")
    del_all["delivery_pct"] = del_all["delivery_pct"].astype("float32")

    # Sort and calculate rolling delivery metrics per symbol
    del_all = del_all.sort_values(["symbol", "date"]).reset_index(drop=True)
    del_all["delivery_vol_20d_ma"] = del_all.groupby("symbol")["delivery_qty"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    del_all["delivery_pct_20d_ma"] = del_all.groupby("symbol")["delivery_pct"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    del_all["abnormal_delivery_ratio"] = np.where(
        del_all["delivery_vol_20d_ma"] > 0,
        del_all["delivery_qty"] / del_all["delivery_vol_20d_ma"],
        1.0,
    ).astype("float32")
    del_all["delivery_pct_spread"] = (del_all["delivery_pct"] - del_all["delivery_pct_20d_ma"]).astype("float32")

    # Merge onto panel on (symbol, date)
    del_features = del_all[["symbol", "date", "delivery_pct", "abnormal_delivery_ratio", "delivery_pct_spread"]].copy()
    del del_all
    gc.collect()

    panel = pd.merge(
        panel,
        del_features,
        on=["symbol", "date"],
        how="left",
    )
    del del_features
    gc.collect()

    # Market structure & liquidity features from panel columns
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    if "turnover" in panel.columns:
        panel["turnover_20d_ma"] = panel.groupby("symbol")["turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        panel["turnover_ratio_20d"] = np.where(panel["turnover_20d_ma"] > 0, panel["turnover"] / panel["turnover_20d_ma"], 1.0)
    else:
        panel["turnover_ratio_20d"] = 1.0

    if "ret_1d" in panel.columns and "turnover" in panel.columns:
        # Amihud illiquidity: (|ret_1d| / turnover) * 1e8
        panel["amihud_illiquidity"] = np.where(
            panel["turnover"] > 0,
            (panel["ret_1d"].abs() / panel["turnover"]) * 1e8,
            np.nan,
        )
    else:
        panel["amihud_illiquidity"] = np.nan

    # Trading suspension & resumption indicators based on calendar session gaps
    panel["prev_date"] = panel.groupby("symbol")["date"].shift(1)
    panel["calendar_gap_days"] = (panel["date"] - panel["prev_date"]).dt.days
    # Normal weekend gap is 3 days. Gaps > 10 calendar days indicate trading suspension
    panel["is_trading_resumption"] = panel["calendar_gap_days"] > 10
    panel["is_prolonged_suspension"] = panel["calendar_gap_days"] > 60

    logger.info("Delivery & market structure features merged successfully.")
    return panel


def merge_point_in_time_ownership(
    panel_df: pd.DataFrame,
    shareholding_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges quarterly shareholding information onto the daily panel strictly as-of
    the filing publication date (merge_asof), ensuring ZERO lookahead leakage.
    """
    logger.info("Merging point-in-time shareholding onto panel (%d rows)...", len(panel_df))
    panel = panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    panel["symbol"] = panel["symbol"].astype(str)

    sh = shareholding_df.copy()
    sh["sh_pub_date"] = pd.to_datetime(sh["publication_date"]).astype("datetime64[ns]")
    sh["symbol"] = sh["symbol"].astype(str)

    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    sh = sh.sort_values(["sh_pub_date", "symbol"]).reset_index(drop=True)

    sh_cols = [
        "promoter_pct", "promoter_change_qoq", "promoter_change_yoy",
        "fii_pct", "fii_change_qoq", "dii_pct", "dii_change_qoq",
        "institutional_pct", "institutional_change_qoq",
        "public_pct", "public_change_qoq", "smart_money_delta"
    ]

    merged = pd.merge_asof(
        panel,
        sh[["symbol", "sh_pub_date"] + sh_cols],
        left_on="date",
        right_on="sh_pub_date",
        by="symbol",
        direction="backward", # Strictly look backwards: filing published on or before date t
    )

    merged["ownership_feature_age_days"] = (merged["date"] - merged["sh_pub_date"]).dt.days
    merged["is_ownership_stale"] = merged["ownership_feature_age_days"] > 120
    merged["has_valid_ownership"] = merged["sh_pub_date"].notna() & (~merged["is_ownership_stale"])

    logger.info("Merged ownership. Valid ownership observations: %d (%.2f%%)",
                int(merged["has_valid_ownership"].sum()),
                float(merged["has_valid_ownership"].sum()) / len(merged) * 100.0)
    return merged
