"""
StockReason v0.9.1 — Panel Feature Engineering & Target Masking Builder.
========================================================================
Builds point-in-time panel features for the 500-stock monthly universe:
1. Ingests split-adjusted prices for eligible point-in-time constituents.
2. Extracts exact 21 features using StockReasonPipelineV091 conventions.
3. Computes forward returns across horizons (1, 5, 21, 63, 126).
4. Masks target windows spanning unadjusted corporate actions (demergers, gaps).
5. Computes cross-sectional demeaned and ranked forward return targets.
6. Saves to data/v0.9.1/features/panel_features.parquet.
"""

import os
import glob
import time
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v091.features.macro import RealMacroDataProvider
from src.v091.canonical.integrity import mask_unadjusted_target_windows
from src.v091.features.targets import construct_cross_sectional_targets

HORIZONS = [1, 5, 21, 63, 126]


def compute_stock_features_and_targets(
    df_stock: pd.DataFrame,
    macro_provider: RealMacroDataProvider,
) -> pd.DataFrame:
    """Computes technical features and forward return targets for a single stock."""
    df = df_stock.sort_values("date").copy()
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    adj_c = df["adj_close"].astype(float)
    c = df["close"].astype(float)
    
    # 1. Trailing Returns (strictly on adj_close)
    df["ret_1d"] = adj_c.pct_change(1)
    df["ret_5d"] = adj_c.pct_change(5)
    df["ret_10d"] = adj_c.pct_change(10)
    df["ret_21d"] = adj_c.pct_change(21)
    df["ret_63d"] = adj_c.pct_change(63)
    df["ret_126d"] = adj_c.pct_change(126)
    
    # 2. Rolling Volatilities (annualized 252 days)
    ret1 = df["ret_1d"]
    df["vol_20d"] = ret1.rolling(20, min_periods=10).std() * np.sqrt(252)
    df["vol_60d"] = ret1.rolling(60, min_periods=30).std() * np.sqrt(252)
    df["vol_60d"] = df["vol_60d"].fillna(df["vol_20d"])
    
    # 3. Distance to Moving Averages (ON ADJ_CLOSE)
    ma20 = adj_c.rolling(20, min_periods=10).mean()
    ma50 = adj_c.rolling(50, min_periods=25).mean()
    ma200 = adj_c.rolling(200, min_periods=100).mean()
    df["dist_ma20"] = (adj_c / (ma20 + 1e-9)) - 1.0
    df["dist_ma50"] = (adj_c / (ma50 + 1e-9)) - 1.0
    df["dist_ma200"] = (adj_c / (ma200 + 1e-9)) - 1.0
    
    # 4. Technical Indicators
    delta = adj_c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    
    # MACD
    ema12 = adj_c.ewm(span=12, adjust=False).mean()
    ema26 = adj_c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = macd_line - signal_line
    
    low_series = df["low"].fillna(c)
    df["hl_spread"] = (df["high"] - df["low"]) / (low_series + 1e-9)
    vol_ma20 = df["volume"].rolling(20, min_periods=10).mean()
    df["vol_ratio_20d"] = df["volume"] / (vol_ma20 + 1.0)
    to_series = df["turnover"].fillna(0.0)
    df["log_turnover"] = np.log1p(to_series.clip(lower=0))
    
    # 5. Forward Return Targets
    for h in HORIZONS:
        # Future price in nominal INR
        df[f"target_price_{h}d"] = c.shift(-h)
        # Future return on adj_close (handles routine corporate splits)
        df[f"target_return_{h}d"] = adj_c.pct_change(h).shift(-h)
        df[f"target_dir_{h}d"] = (df[f"target_return_{h}d"] > 0).astype(float)
        
    return df


def build_v091_panel(
    pit_universe_path: str = "data/v0.9.1/canonical/universe_membership/monthly_pit_universe.parquet",
    output_path: str = "data/v0.9.1/features/panel_features.parquet",
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Builds the complete point-in-time feature panel for v0.9.1.
    """
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load Universe
    liq_df = pd.read_parquet(pit_universe_path)
    symbols = sorted(liq_df["symbol"].unique().tolist())
    if max_symbols:
        symbols = symbols[:max_symbols]
    print(f"Building v0.9.1 panel for {len(symbols)} point-in-time symbols...")
    
    # 2. Ingest Adjusted Prices
    adj_files = sorted(glob.glob("data/indian-markets/prices_adjusted/nse_*.parquet"))
    dfs = []
    for f in adj_files:
        df = pd.read_parquet(f, columns=["date", "symbol", "series", "open", "high", "low", "close", "prev_close", "volume", "turnover", "adj_factor_cumulative", "adj_close"])
        eq = df[(df["series"] == "EQ") & (df["symbol"].isin(symbols))]
        dfs.append(eq)
    full_prices = pd.concat(dfs, ignore_index=True)
    full_prices["date_str"] = pd.to_datetime(full_prices["date"]).dt.strftime("%Y-%m-%d")
    
    # 3. Macro Provider
    macro = RealMacroDataProvider()
    
    # 4. Feature Extraction per stock
    processed = []
    for sym, group in full_prices.groupby("symbol"):
        if len(group) >= 30:
            stk_feat = compute_stock_features_and_targets(group, macro)
            processed.append(stk_feat)
    panel = pd.concat(processed, ignore_index=True)
    
    # 5. Merge Macro Features
    macro_records = []
    for d in panel["date_str"].unique():
        try:
            m_vec = macro.get_macro_vector(d)
            m_vec["date_str"] = d
            macro_records.append(m_vec)
        except Exception:
            pass
    df_macro = pd.DataFrame(macro_records)
    panel = pd.merge(panel, df_macro, on="date_str", how="left")
    panel["rel_strength_21d"] = panel["ret_21d"] - panel["nifty_ret_21d"]
    
    # 6. Mask Unadjusted Corporate Action Windows (Task 1b)
    print("Masking unadjusted corporate action windows (demergers, gaps)...")
    panel, mask_counts = mask_unadjusted_target_windows(panel, horizons=HORIZONS)
    print(f"Masked counts by horizon: {mask_counts}")
    
    # 7. Construct Cross-Sectional Demeaned Targets (Task 4)
    print("Constructing cross-sectional demeaned targets...")
    panel = construct_cross_sectional_targets(panel, horizons=HORIZONS)
    
    panel.to_parquet(output_path, index=False)
    print(f"Panel saved to {output_path}: {len(panel):,} rows across {panel['symbol'].nunique()} equities in {time.time() - start_time:.1f}s")
    return panel
