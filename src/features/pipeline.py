"""Canonical Feature Assembly Pipeline."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from src.features.registry import DETAILED_FEATURES
from src.features.technical import compute_technical_features
from src.features.market import merge_market_and_sector_features, load_market_benchmark_indices
from src.features.fundamental import merge_point_in_time_financials, load_and_clean_annual_fundamentals
from src.features.ownership import merge_point_in_time_ownership, load_clean_shareholding_data, load_delivery_features_for_panel
from src.features.liquidity import compute_liquidity_features

logger = logging.getLogger("feature_pipeline")

POPULATED_FEATURE_NAMES = [
    # F0 (17)
    "close", "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d",
    "vol_20d", "vol_60d", "dist_ma20", "dist_ma50", "dist_ma200", "rsi_14",
    "macd_diff", "hl_spread", "vol_ratio_20d", "log_turnover",
    # F1 (7)
    "nifty50_ret_1d", "nifty50_ret_5d", "nifty50_ret_21d", "nifty50_vol_20d",
    "india_vix_level", "india_vix_change_5d", "market_breadth_pct_up",
    # F2 (5)
    "excess_ret_market_1d", "excess_ret_market_21d", "cross_sec_volume_pctile",
    "cross_sec_turnover_pctile", "cross_sec_vol_pctile",
    # F3 (20)
    "rev_growth_yoy", "rev_acceleration", "pat_growth_yoy", "eps_growth_yoy",
    "ebit_growth_yoy", "operating_margin", "net_profit_margin", "gross_margin",
    "margin_expansion_yoy", "debt_to_equity", "interest_coverage", "current_ratio",
    "debt_growth_yoy", "free_cash_flow", "capex_intensity", "cash_conversion",
    "profit_to_fcf_divergence", "earnings_quality", "pe_ratio", "earnings_yield",
    # F6 (19)
    "promoter_pct", "promoter_change_qoq", "promoter_change_yoy", "fii_pct",
    "fii_change_qoq", "dii_pct", "dii_change_qoq", "institutional_pct",
    "institutional_change_qoq", "public_pct", "public_change_qoq", "smart_money_delta",
    "delivery_pct", "abnormal_delivery_ratio", "delivery_pct_spread", "turnover_20d_ma",
    "turnover_ratio_20d", "amihud_illiquidity", "is_trading_resumption", "is_prolonged_suspension"
]


def assemble_canonical_features(
    base_panel_df: pd.DataFrame,
    indices_dir: str = "data/external/sector_indices",
    financials_path: str = "data/external/company_financials/Phddata/india_annual_fundamentals_2010_2026.parquet",
    shareholding_dir: str = "data/external/shareholding/bse_xbrl_2015_2026",
    delivery_dir: str = "data/indian-markets/delivery",
) -> pd.DataFrame:
    """
    Assembles the canonical feature dataset by merging:
    - F0: Technical features
    - F1 & F2: Market benchmarks & cross-sectional percentiles
    - F3: Annual fundamentals (point-in-time)
    - F6: Ownership & delivery dynamics
    """
    logger.info("Starting canonical feature panel assembly on %d rows...", len(base_panel_df))
    df = base_panel_df.copy()

    # 1. Market & Sector features
    if Path(indices_dir).exists():
        market_df = load_market_benchmark_indices(indices_dir)
        if len(market_df) > 0:
            df = merge_market_and_sector_features(df, market_df)

    # 2. Fundamentals
    if Path(financials_path).exists():
        fin_df = load_and_clean_annual_fundamentals(financials_path)
        if len(fin_df) > 0:
            df = merge_point_in_time_financials(df, fin_df)

    # 3. Ownership & Delivery
    if Path(shareholding_dir).exists():
        sh_df = load_clean_shareholding_data(shareholding_dir)
        if len(sh_df) > 0:
            df = merge_point_in_time_ownership(df, sh_df)

    if Path(delivery_dir).exists():
        deliv_df = load_delivery_features_for_panel(delivery_dir)
        if len(deliv_df) > 0:
            df = pd.merge(df, deliv_df, on=["symbol", "date"], how="left")

    logger.info("Completed feature assembly. Final shape: %s", df.shape)
    return df
