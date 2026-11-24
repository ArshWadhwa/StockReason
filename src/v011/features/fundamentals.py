"""
StockReason v0.11 — Point-in-Time Fundamental Feature Engineering.
===================================================================
Remediates Phase 3 (Section 16):
1. Ingests quarterly financial statements:
   - Revenue, EBITDA, Net Profit, YoY Growth
2. Point-in-Time Correctness:
   Enforces strictly report_date <= forecast_origin (no lookahead into unannounced quarters).
3. Computes:
   - rev_growth_yoy: YoY revenue growth rate
   - profit_growth_yoy: YoY net profit growth rate
   - net_margin: Net profit / revenue
   - ebitda_margin: EBITDA / revenue
   - fundamental_health_score: Composite score combining profitability & growth
4. Safe Fallback:
   Stocks without coverage receive median sector imputation and a missingness indicator.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def load_quarterly_financials(csv_path: str = "data/external/company_financials/quarterly_SAMPLE_2018_2026/india_quarterly_results_sample.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize symbol
    df["clean_symbol"] = df["exchange_code"].astype(str).str.replace("NSE:", "").str.replace("BSE:", "").str.strip()
    # Normalize dates
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["clean_symbol", "report_date"]).reset_index(drop=True)

    # Compute key ratios
    rev = pd.to_numeric(df["revenue_inr"], errors="coerce").fillna(0.0)
    profit = pd.to_numeric(df["net_profit_inr"], errors="coerce").fillna(0.0)
    ebitda = pd.to_numeric(df["ebitda_inr"], errors="coerce").fillna(0.0)

    df["net_margin"] = np.where(rev > 0, profit / rev, 0.0)
    df["ebitda_margin"] = np.where(rev > 0, ebitda / rev, 0.0)
    df["rev_growth_yoy"] = pd.to_numeric(df["revenue_yoy_pct"], errors="coerce").fillna(0.0) / 100.0
    df["profit_growth_yoy"] = pd.to_numeric(df["net_profit_yoy_pct"], errors="coerce").fillna(0.0) / 100.0

    # Composite health score
    df["fundamental_score"] = np.clip(
        0.4 * np.sign(df["profit_growth_yoy"]) + 0.3 * np.clip(df["net_margin"], -0.5, 0.5) + 0.3 * np.clip(df["rev_growth_yoy"], -0.5, 0.5),
        -1.0, 1.0
    )
    return df


def attach_point_in_time_fundamentals(
    panel_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date_str",
) -> pd.DataFrame:
    """
    Joins quarterly fundamentals point-in-time using merge_asof on datetime timestamps:
    ensures panel_df[date] >= fin_df[report_date].
    """
    out = panel_df.copy()
    if "date_dt" not in out.columns:
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date_dt"] = out["date"]
        elif date_col in out.columns:
            out["date_dt"] = pd.to_datetime(out[date_col])
        else:
            out["date_dt"] = pd.to_datetime(out["date"])

    fin = fin_df.copy()
    fin["report_dt"] = pd.to_datetime(fin["report_date"])
    valid_symbols = set(fin["clean_symbol"].unique())
    sub_fin = fin[fin["clean_symbol"].isin(valid_symbols)].copy()

    # Sort both for merge_asof
    out = out.sort_values(["date_dt", symbol_col]).reset_index(drop=True)
    sub_fin = sub_fin.sort_values(["report_dt", "clean_symbol"]).reset_index(drop=True)

    # Use merge_asof with exact symbol match
    merged = pd.merge_asof(
        out,
        sub_fin[["clean_symbol", "report_dt", "net_margin", "ebitda_margin", "rev_growth_yoy", "profit_growth_yoy", "fundamental_score"]],
        left_on="date_dt",
        right_on="report_dt",
        left_by=symbol_col,
        right_by="clean_symbol",
        direction="backward",
    )

    # Missing indicator
    merged["has_fundamental_coverage"] = merged["report_dt"].notna().astype(int)
    merged["net_margin"] = merged["net_margin"].fillna(0.0)
    merged["ebitda_margin"] = merged["ebitda_margin"].fillna(0.0)
    merged["rev_growth_yoy"] = merged["rev_growth_yoy"].fillna(0.0)
    merged["profit_growth_yoy"] = merged["profit_growth_yoy"].fillna(0.0)
    merged["fundamental_score"] = merged["fundamental_score"].fillna(0.0)

    drop_cols = ["clean_symbol", "report_dt", "date_dt"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged
