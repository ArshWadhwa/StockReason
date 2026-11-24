"""
StockReason v0.9.2 — External Data Ingestion (Annual Financials & Sector Indices).
==================================================================================
Remediates Item 8:
- Ingests annual balance sheets and income statements from Phddata downloads.
- Preserves publication-time correctness:
  Annual accounts for FY ending March 31 are strictly lagged to October 1 of that year
  (6-month reporting lag, preventing lookahead).
- Ingests sector indices (Nifty Bank, Nifty IT, Nifty Pharma, etc.) and computes
  sector momentum and sector-relative returns.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def load_annual_financial_ratios(
    fin_core_path: str = "data/external/company_financials/Phddata/downloads/fin_core_wide.csv.gz",
    publication_lag_months: int = 6, # Available on Oct 1 for March 31 fiscal year end
) -> pd.DataFrame:
    """
    Ingests and computes annual financial quality ratios with strict publication lag.
    """
    if not os.path.exists(fin_core_path):
        return pd.DataFrame()

    cols_to_read = [
        "symbol", "fy_end_year", "revenue_from_operations", "total_income",
        "total_equity", "equity_share_capital", "other_equity",
        "profit_for_year", "profit_before_tax", "total_assets"
    ]
    # Check which columns exist in file
    sample = pd.read_csv(fin_core_path, nrows=5)
    valid_cols = [c for c in cols_to_read if c in sample.columns]

    df_fin = pd.read_csv(fin_core_path, usecols=valid_cols)
    df_fin = df_fin.dropna(subset=["symbol", "fy_end_year"])
    df_fin["fy_end_year"] = pd.to_numeric(df_fin["fy_end_year"], errors="coerce").dropna().astype(int)

    # Compute key ratios robustly
    if "profit_for_year" in df_fin.columns:
        pat = pd.to_numeric(df_fin["profit_for_year"], errors="coerce").fillna(0.0)
    elif "profit_before_tax" in df_fin.columns:
        pat = pd.to_numeric(df_fin["profit_before_tax"], errors="coerce").fillna(0.0)
    else:
        pat = pd.Series(0.0, index=df_fin.index)

    if "revenue_from_operations" in df_fin.columns:
        rev = pd.to_numeric(df_fin["revenue_from_operations"], errors="coerce").fillna(0.0)
    elif "total_income" in df_fin.columns:
        rev = pd.to_numeric(df_fin["total_income"], errors="coerce").fillna(0.0)
    else:
        rev = pd.Series(0.0, index=df_fin.index)

    if "total_equity" in df_fin.columns:
        total_eq = pd.to_numeric(df_fin["total_equity"], errors="coerce").fillna(0.0)
    else:
        eq_share = pd.to_numeric(df_fin.get("equity_share_capital", 0.0), errors="coerce").fillna(0.0)
        oth_eq = pd.to_numeric(df_fin.get("other_equity", 0.0), errors="coerce").fillna(0.0)
        total_eq = eq_share + oth_eq

    if "total_assets" in df_fin.columns:
        assets = pd.to_numeric(df_fin["total_assets"], errors="coerce").fillna(0.0)
    else:
        assets = pd.Series(0.0, index=df_fin.index)

    df_fin["profit_margin"] = np.where(rev.abs() > 1.0, pat / (rev + 1e-4), 0.0)
    df_fin["roe"] = np.where(total_eq.abs() > 1.0, pat / (total_eq + 1e-4), 0.0)
    df_fin["asset_turnover"] = np.where(assets > 1.0, rev / (assets + 1e-4), 0.0)

    # Enforce publication-time availability date:
    # Fiscal year ending in YYYY is published and known by October 1 of year YYYY
    df_fin["available_from_date"] = df_fin["fy_end_year"].astype(str) + "-10-01"

    out_cols = ["symbol", "fy_end_year", "available_from_date", "profit_margin", "roe", "asset_turnover"]
    return df_fin[out_cols].sort_values(["symbol", "available_from_date"]).drop_duplicates(subset=["symbol", "available_from_date"])


def load_sector_indices_features(
    sector_dir: str = "data/external/sector_indices",
) -> Dict[str, pd.DataFrame]:
    """
    Ingests sector indices (IT, Bank, Pharma, 100, 200, 500) and computes 21d/63d returns.
    """
    files = glob.glob(os.path.join(sector_dir, "*.parquet"))
    sector_dfs = {}
    for f in files:
        base = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns or "date" in df.columns:
                    col = "Date" if "Date" in df.columns else "date"
                    df = df.set_index(pd.to_datetime(df[col]))
            df = df.sort_index()
            # Compute returns
            close_col = "Close" if "Close" in df.columns else "close"
            if close_col in df.columns:
                c = df[close_col].astype(float)
                ret_21d = c.pct_change(21)
                ret_63d = c.pct_change(63)
                feat_df = pd.DataFrame({
                    "date_str": df.index.strftime("%Y-%m-%d"),
                    f"{base.lower()}_ret_21d": ret_21d.values,
                    f"{base.lower()}_ret_63d": ret_63d.values
                }).dropna()
                sector_dfs[base] = feat_df
        except Exception:
            pass
    return sector_dfs
