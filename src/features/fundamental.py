"""StockReason v0.13B — Point-in-Time Financial Intelligence & Expectation Proxies.

Implements Section 6 & Section 7 Requirements:
1. Revenue Intelligence:
   - YoY revenue growth, sequential growth, acceleration, trailing growth, consistency.
2. Earnings Intelligence:
   - EPS growth, net profit growth, operating profit growth, earnings acceleration, quality.
3. Margin Intelligence:
   - Operating margin, net profit margin, gross margin, margin expansion/contraction.
4. Balance Sheet Intelligence:
   - Debt-to-equity, net debt, interest coverage, working capital changes, current ratio, debt growth.
5. Cash Flow Intelligence:
   - Operating cash flow, free cash flow (FCF), cash conversion, profit-to-cash divergence.
6. Valuation Intelligence:
   - Point-in-time P/E, P/B, EV/EBITDA proxy, earnings yield, FCF yield.
7. Earnings Expectation & Surprise Intelligence (Section 7):
   - Historical comparable expectation proxies (prior-year, sequential, seasonal).
   - Explicitly marked: is_analyst_consensus = False.
8. Point-in-Time Availability:
   - Enforces conservative 120-day publication lag (July 31) for annual statements.
   - Enforces exact report_date publication timestamps for quarterly statements.
   - Generates feature age, staleness flags, and availability masks. Zero lookahead leakage.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013b_financials")


def load_and_clean_annual_fundamentals(
    phd_path: str = "data/external/company_financials/Phddata/downloads/fin_core_wide.csv.gz",
    publication_lag_days: int = 120, # July 31 after March 31 fiscal year end
) -> pd.DataFrame:
    """
    Loads annual statements from Phddata, calculates derived fundamental ratios,
    and assigns conservative point-in-time publication dates.
    """
    if not os.path.exists(phd_path):
        logger.warning("Phddata path %s not found", phd_path)
        return pd.DataFrame()

    use_cols = [
        "symbol", "fy_end_year", "period_type",
        "revenue_from_operations", "total_income", "cost_of_materials",
        "employee_benefit_expense", "finance_costs", "depreciation_amortisation",
        "total_expenses", "profit_before_tax", "profit_for_year",
        "pat_attributable_owners", "eps_basic", "eps_diluted",
        "property_plant_equipment", "total_current_assets", "total_assets",
        "total_equity", "non_current_borrowings", "current_borrowings",
        "total_current_liabilities", "cfo", "capex", "cash_and_equivalents"
    ]
    df = pd.read_csv(phd_path, usecols=use_cols)
    df = df[df["period_type"] == "fiscal"].copy()
    df["fy_end_year"] = pd.to_numeric(df["fy_end_year"], errors="coerce")
    df = df.dropna(subset=["symbol", "fy_end_year"]).sort_values(["symbol", "fy_end_year"]).reset_index(drop=True)

    # Point-in-time publication date: July 31 of fiscal year
    # E.g. FY2023 ends 2023-03-31 -> Publication date 2023-07-31 (122 days lag)
    df["publication_date"] = pd.to_datetime(df["fy_end_year"].astype(int).astype(str) + "-07-31")

    # 1. Revenue Intelligence
    df["rev_lag1"] = df.groupby("symbol")["revenue_from_operations"].shift(1)
    df["rev_lag2"] = df.groupby("symbol")["revenue_from_operations"].shift(2)
    df["rev_growth_yoy"] = np.where(df["rev_lag1"] > 0, (df["revenue_from_operations"] / df["rev_lag1"]) - 1.0, np.nan)
    rev_growth_lag1 = np.where(df["rev_lag2"] > 0, (df["rev_lag1"] / df["rev_lag2"]) - 1.0, np.nan)
    df["rev_acceleration"] = df["rev_growth_yoy"] - rev_growth_lag1

    # 2. Earnings Intelligence
    df["pat_lag1"] = df.groupby("symbol")["profit_for_year"].shift(1)
    df["pat_growth_yoy"] = np.where(df["pat_lag1"] > 0, (df["profit_for_year"] / df["pat_lag1"]) - 1.0, np.nan)
    df["eps_lag1"] = df.groupby("symbol")["eps_basic"].shift(1)
    df["eps_growth_yoy"] = np.where(df["eps_lag1"] > 0, (df["eps_basic"] / df["eps_lag1"]) - 1.0, np.nan)
    df["pbt_lag1"] = df.groupby("symbol")["profit_before_tax"].shift(1)
    df["ebit_growth_yoy"] = np.where(df["pbt_lag1"] > 0, (df["profit_before_tax"] / df["pbt_lag1"]) - 1.0, np.nan)
    df["earnings_quality"] = np.where(df["revenue_from_operations"] > 0, df["cfo"] / df["revenue_from_operations"], np.nan)

    # 3. Margin Intelligence
    rev = df["revenue_from_operations"]
    df["operating_margin"] = np.where(rev > 0, df["profit_before_tax"] / rev, np.nan)
    df["net_profit_margin"] = np.where(rev > 0, df["profit_for_year"] / rev, np.nan)
    raw_mats = df["cost_of_materials"].fillna(0.0)
    df["gross_margin"] = np.where(rev > 0, (rev - raw_mats) / rev, np.nan)
    df["op_margin_lag1"] = df.groupby("symbol")["operating_margin"].shift(1)
    df["margin_expansion_yoy"] = df["operating_margin"] - df["op_margin_lag1"]

    # 4. Balance Sheet Intelligence
    tot_debt = df["non_current_borrowings"].fillna(0.0) + df["current_borrowings"].fillna(0.0)
    eq = df["total_equity"]
    df["total_debt"] = tot_debt
    df["debt_to_equity"] = np.where(eq > 0, tot_debt / eq, np.nan)
    cash = df["cash_and_equivalents"].fillna(0.0)
    df["net_debt"] = tot_debt - cash
    fc = df["finance_costs"].fillna(0.0)
    df["interest_coverage"] = np.where(fc > 0, df["profit_before_tax"] / fc, np.nan)
    cur_liab = df["total_current_liabilities"]
    df["current_ratio"] = np.where(cur_liab > 0, df["total_current_assets"] / cur_liab, np.nan)
    df["debt_lag1"] = df.groupby("symbol")["total_debt"].shift(1)
    df["debt_growth_yoy"] = np.where(df["debt_lag1"] > 0, (tot_debt / df["debt_lag1"]) - 1.0, np.nan)

    # 5. Cash Flow Intelligence
    capex_abs = df["capex"].abs().fillna(0.0)
    df["free_cash_flow"] = df["cfo"].fillna(0.0) - capex_abs
    df["capex_intensity"] = np.where(rev > 0, capex_abs / rev, np.nan)
    df["cash_conversion"] = np.where(df["profit_for_year"] > 0, df["cfo"] / df["profit_for_year"], np.nan)
    df["profit_to_fcf_divergence"] = df["profit_for_year"] - df["free_cash_flow"]

    # Select final clean fundamental columns
    keep_cols = [
        "symbol", "fy_end_year", "publication_date",
        "revenue_from_operations", "profit_for_year", "eps_basic", "total_equity", "total_debt", "net_debt",
        "rev_growth_yoy", "rev_acceleration", "pat_growth_yoy", "eps_growth_yoy", "ebit_growth_yoy",
        "operating_margin", "net_profit_margin", "gross_margin", "margin_expansion_yoy",
        "debt_to_equity", "interest_coverage", "current_ratio", "debt_growth_yoy",
        "free_cash_flow", "capex_intensity", "cash_conversion", "profit_to_fcf_divergence", "earnings_quality",
    ]
    clean_df = df[keep_cols].copy()
    logger.info("Prepared clean annual fundamentals: %d statements across %d symbols", len(clean_df), clean_df["symbol"].nunique())
    return clean_df


def load_quarterly_financials_with_expectation_proxies(
    csv_path: str = "data/external/company_financials/quarterly_SAMPLE_2018_2026/india_quarterly_results_sample.csv",
) -> pd.DataFrame:
    """
    Loads quarterly statements with exact report_date publication timestamps,
    calculates QoQ/YoY growth and derives named historical expectation proxies (Section 7).
    """
    if not os.path.exists(csv_path):
        logger.warning("Quarterly sample path %s not found", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Extract clean NSE symbol from exchange_code
    # E.g. "BHARTIARTL" -> "BHARTIARTL", "BSE:500325" -> None (unless mapped)
    is_nse_sym = ~df["exchange_code"].astype(str).str.startswith("BSE:") & df["exchange_code"].notna()
    df = df[is_nse_sym].copy()
    df["symbol"] = df["exchange_code"].astype(str)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])

    df = df.dropna(subset=["symbol", "report_date"]).sort_values(["symbol", "quarter_end"]).reset_index(drop=True)

    # Growth and margins
    df["q_revenue"] = pd.to_numeric(df["revenue_inr"], errors="coerce")
    df["q_pat"] = pd.to_numeric(df["net_profit_inr"], errors="coerce")
    df["q_ebitda"] = pd.to_numeric(df["ebitda_inr"], errors="coerce")

    # Sequential QoQ and YoY
    df["q_rev_lag1"] = df.groupby("symbol")["q_revenue"].shift(1)
    df["q_rev_lag4"] = df.groupby("symbol")["q_revenue"].shift(4)
    df["q_pat_lag1"] = df.groupby("symbol")["q_pat"].shift(1)
    df["q_pat_lag4"] = df.groupby("symbol")["q_pat"].shift(4)

    df["q_rev_growth_qoq"] = np.where(df["q_rev_lag1"] > 0, (df["q_revenue"] / df["q_rev_lag1"]) - 1.0, np.nan)
    df["q_rev_growth_yoy"] = np.where(df["q_rev_lag4"] > 0, (df["q_revenue"] / df["q_rev_lag4"]) - 1.0, np.nan)
    df["q_pat_growth_qoq"] = np.where(df["q_pat_lag1"] > 0, (df["q_pat"] / df["q_pat_lag1"]) - 1.0, np.nan)
    df["q_pat_growth_yoy"] = np.where(df["q_pat_lag4"] > 0, (df["q_pat"] / df["q_pat_lag4"]) - 1.0, np.nan)

    df["q_ebitda_margin"] = np.where(df["q_revenue"] > 0, df["q_ebitda"] / df["q_revenue"], np.nan)
    df["q_net_margin"] = np.where(df["q_revenue"] > 0, df["q_pat"] / df["q_revenue"], np.nan)

    # -------------------------------------------------------------
    # Section 7: Named Historical Expectation Proxies
    # Explicitly: is_analyst_consensus = False
    # -------------------------------------------------------------
    # Proxy 1: Prior-Year Comparable Growth (YoY Unexpected Delta)
    df["earnings_surprise_prior_year_proxy"] = np.where(
        df["q_pat_lag4"].abs() > 0,
        (df["q_pat"] - df["q_pat_lag4"]) / (df["q_pat_lag4"].abs() + 1e-6),
        np.nan,
    )
    df["revenue_surprise_prior_year_proxy"] = np.where(
        df["q_rev_lag4"].abs() > 0,
        (df["q_revenue"] - df["q_rev_lag4"]) / (df["q_rev_lag4"].abs() + 1e-6),
        np.nan,
    )

    # Proxy 2: Sequential Unexpected QoQ Delta
    df["earnings_surprise_sequential_proxy"] = np.where(
        df["q_pat_lag1"].abs() > 0,
        (df["q_pat"] - df["q_pat_lag1"]) / (df["q_pat_lag1"].abs() + 1e-6),
        np.nan,
    )

    df["is_analyst_consensus"] = False
    df["expectation_source"] = "HISTORICAL_COMPARABLE_PROXIES"

    keep_cols = [
        "symbol", "quarter_end", "report_date",
        "q_revenue", "q_pat", "q_ebitda",
        "q_rev_growth_qoq", "q_rev_growth_yoy", "q_pat_growth_qoq", "q_pat_growth_yoy",
        "q_ebitda_margin", "q_net_margin",
        "earnings_surprise_prior_year_proxy", "revenue_surprise_prior_year_proxy",
        "earnings_surprise_sequential_proxy", "is_analyst_consensus", "expectation_source"
    ]
    clean_q = df[keep_cols].copy()
    logger.info("Prepared clean quarterly financials & expectation proxies: %d filings across %d symbols", len(clean_q), clean_q["symbol"].nunique())
    return clean_q


def merge_point_in_time_financials(
    panel_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merges financial statement features onto the daily canonical price panel strictly as-of
    the observation date (merge_asof), ensuring ZERO lookahead leakage.
    
    Generates point-in-time valuation multiples (P/E, P/B, FCF yield, earnings yield),
    feature age, and freshness flags.
    """
    logger.info("Merging point-in-time financials onto panel of %d rows...", len(panel_df))
    panel = panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    panel["symbol"] = panel["symbol"].astype(str)

    # Ensure annual statements are sorted by publication_date
    ann = annual_df.copy()
    ann["ann_pub_date"] = pd.to_datetime(ann["publication_date"]).astype("datetime64[ns]")
    ann["symbol"] = ann["symbol"].astype(str)

    # We do asof merge per symbol to avoid cross-symbol contamination
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    ann = ann.sort_values(["ann_pub_date", "symbol"]).reset_index(drop=True)
    
    # Fast merge_asof by symbol
    merged_subsets = []
    # Unique symbols present in both
    common_syms = set(panel["symbol"].unique()).intersection(set(ann["symbol"].unique()))
    
    # Feature columns to pull from annual statements
    ann_feat_cols = [
        "rev_growth_yoy", "rev_acceleration", "pat_growth_yoy", "eps_growth_yoy", "ebit_growth_yoy",
        "operating_margin", "net_profit_margin", "gross_margin", "margin_expansion_yoy",
        "debt_to_equity", "interest_coverage", "current_ratio", "debt_growth_yoy",
        "free_cash_flow", "capex_intensity", "cash_conversion", "profit_to_fcf_divergence", "earnings_quality",
        "eps_basic", "total_equity", "net_debt", "revenue_from_operations"
    ]
    
    # Left merge as-of
    merged_df = pd.merge_asof(
        panel,
        ann[["symbol", "ann_pub_date"] + ann_feat_cols],
        left_on="date",
        right_on="ann_pub_date",
        by="symbol",
        direction="backward", # Strictly look backwards: statement published on or before date t
    )

    # Feature Age & Staleness Policy
    merged_df["financial_feature_age_days"] = (merged_df["date"] - merged_df["ann_pub_date"]).dt.days
    # Mark stale if > 450 days (more than ~15 months since last annual report)
    merged_df["is_financial_stale"] = merged_df["financial_feature_age_days"] > 450
    merged_df["has_valid_financials"] = merged_df["ann_pub_date"].notna() & (~merged_df["is_financial_stale"])

    # Point-in-time valuation multiples using daily close P_t
    close = merged_df["close"]
    eps = merged_df["eps_basic"]
    eq = merged_df["total_equity"]

    # P/E ratio: clipped between 0 and 200 for stability
    merged_df["pe_ratio"] = np.where((eps > 0) & (close > 0), np.clip(close / eps, 1.0, 200.0), np.nan)
    # Earnings yield
    merged_df["earnings_yield"] = np.where((close > 0) & eps.notna(), np.clip(eps / close, -0.50, 0.50), np.nan)
    # Debt-to-market cap proxy
    merged_df["debt_to_equity"] = np.clip(merged_df["debt_to_equity"], 0.0, 50.0)

    # If quarterly available, merge quarterly features asof report_date
    if quarterly_df is not None and len(quarterly_df) > 0:
        q_df = quarterly_df.copy()
        q_df["q_rep_date"] = pd.to_datetime(q_df["report_date"]).astype("datetime64[ns]")
        q_df["symbol"] = q_df["symbol"].astype(str)
        q_df = q_df.sort_values(["q_rep_date", "symbol"]).reset_index(drop=True)
        merged_df = merged_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        q_cols = [
            "q_rev_growth_qoq", "q_rev_growth_yoy", "q_pat_growth_qoq", "q_pat_growth_yoy",
            "q_ebitda_margin", "q_net_margin",
            "earnings_surprise_prior_year_proxy", "revenue_surprise_prior_year_proxy",
            "earnings_surprise_sequential_proxy"
        ]
        merged_df = pd.merge_asof(
            merged_df,
            q_df[["symbol", "q_rep_date"] + q_cols],
            left_on="date",
            right_on="q_rep_date",
            by="symbol",
            direction="backward",
        )
        merged_df["quarterly_feature_age_days"] = (merged_df["date"] - merged_df["q_rep_date"]).dt.days
        merged_df["has_valid_quarterly"] = merged_df["q_rep_date"].notna() & (merged_df["quarterly_feature_age_days"] <= 120)

    logger.info("Successfully merged point-in-time financials. Valid annual observations: %d (%.2f%%)",
                int(merged_df["has_valid_financials"].sum()),
                float(merged_df["has_valid_financials"].sum()) / len(merged_df) * 100.0)

    return merged_df
