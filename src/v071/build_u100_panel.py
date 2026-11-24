"""
StockReason v0.7.1 - U100 Panel Builder
=======================================
Constructs the full U100 panel dataset from data/v0.7/raw/cleaned_stock_panel.parquet
and data/v0.7/raw/INDEX_NSEI.parquet with Feature Set C features and multi-horizon targets.
"""

import os
import numpy as np
import pandas as pd
from src.v07.features.targets import compute_targets
from src.v07.features.feature_sets import enrich_stationary_features

def build_u100_panel():
    raw_path = "data/v0.7/raw/cleaned_stock_panel.parquet"
    bench_path = "data/v0.7/raw/INDEX_NSEI.parquet"
    out_path = "data/v0.7.1/panel_u100.parquet"
    os.makedirs("data/v0.7.1", exist_ok=True)

    print(f"Loading raw stock panel from {raw_path}...")
    df_raw = pd.read_parquet(raw_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.tz_localize(None)
    
    print(f"Loading benchmark index from {bench_path}...")
    df_bench = pd.read_parquet(bench_path)
    df_bench["date"] = pd.to_datetime(df_bench["date"]).dt.tz_localize(None)
    bench_map = df_bench.set_index("date")["close"].to_dict()

    df_raw["nifty_50_close"] = df_raw["date"].map(bench_map)
    # Forward fill any missing holiday/weekend benchmark values
    df_raw["nifty_50_close"] = df_raw["nifty_50_close"].ffill().bfill()

    print(f"Loaded {len(df_raw)} rows across {df_raw['ticker'].nunique()} tickers.")
    print("Computing technical features per ticker...")

    feature_dfs = []
    for ticker, group in df_raw.groupby("ticker"):
        g = group.sort_values("date").copy()
        close = g["close"]
        adj_close = g["adj_close"]
        high = g["high"]
        low = g["low"]
        vol = g["volume"].fillna(0)

        # Returns
        for p in [1, 2, 3, 5, 10, 15, 21, 42]:
            g[f"return_{p}d"] = (adj_close - adj_close.shift(p)) / (adj_close.shift(p) + 1e-9)
            g[f"log_return_{p}d"] = np.log(adj_close / (adj_close.shift(p) + 1e-9))

        # Realized volatility
        ret_1d = g["return_1d"]
        g["vol_20d"] = ret_1d.rolling(20).std() * np.sqrt(252)
        g["vol_60d"] = ret_1d.rolling(60).std() * np.sqrt(252)

        # ATR / NATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["natr_14"] = (tr.rolling(14).mean() / (close + 1e-9)) * 100.0
        g["natr_20"] = (tr.rolling(20).mean() / (close + 1e-9)) * 100.0

        # Moving average distance
        for ma in [10, 20, 50, 100, 200]:
            sma = close.rolling(ma).mean()
            g[f"dist_sma_{ma}"] = (close - sma) / (sma + 1e-9)
            g[f"close_to_sma{ma}"] = g[f"dist_sma_{ma}"]

        # RSI
        for r_period in [14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(r_period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(r_period).mean()
            rs = gain / (loss + 1e-9)
            g[f"rsi_{r_period}"] = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2.0 * std20
        lower = sma20 - 2.0 * std20
        g["bb_pct_b_20"] = (close - lower) / (upper - lower + 1e-9)
        g["bb_width_20"] = (upper - lower) / (sma20 + 1e-9)

        sma50 = close.rolling(50).mean()
        std50 = close.rolling(50).std()
        g["bb_width_50"] = (4.0 * std50) / (sma50 + 1e-9)
        g["sma_50_200_ratio"] = sma50 / (close.rolling(200).mean() + 1e-9)

        # Volume ratio
        vol_sma20 = vol.rolling(20).mean()
        g["vol_ratio_20"] = vol / (vol_sma20 + 1e-9)
        g["high_low_spread"] = (high - low) / (close + 1e-9)

        feature_dfs.append(g)

    df_feats = pd.concat(feature_dfs, ignore_index=True)
    print("Computing multi-horizon stationary excess return targets...")
    df_targets = compute_targets(df_feats, horizons=(5, 21), price_col="adj_close", benchmark_col="nifty_50_close")

    print(f"Saving final U100 panel to {out_path}...")
    df_targets.to_parquet(out_path, index=False)
    print(f"Panel saved successfully. Shape: {df_targets.shape}, Dates: {df_targets['date'].nunique()}, Tickers: {df_targets['ticker'].nunique()}")

if __name__ == "__main__":
    build_u100_panel()
