import logging
from pathlib import Path
import pandas as pd
import yaml

from .fetcher import fetch_multiple_stocks, fetch_all_indices, fetch_all_macro
from .cleaner import align_and_clean_dataset
from .technical_features import generate_technical_features
from .lag_features import generate_lag_features
from .regime_detection import detect_market_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_configs():
    with open(BASE_DIR / "config" / "tickers.yaml", "r") as f:
        tickers_cfg = yaml.safe_load(f)
    with open(BASE_DIR / "config" / "settings.yaml", "r") as f:
        settings_cfg = yaml.safe_load(f)
    return tickers_cfg, settings_cfg

def build_pipeline():
    """
    Master pipeline orchestrator (Week 1 -> Week 3):
    Fetch Raw Data -> Clean & Align -> Merge Market Context -> 100+ Features -> Lag Features -> Regime Detection -> Export price_features.parquet
    """
    logger.info("==================================================")
    logger.info("Starting M1 Feature Pipeline Execution (Week 1 to Week 3)")
    logger.info("==================================================")

    tickers_cfg, settings_cfg = load_configs()

    period = settings_cfg.get("fetcher", {}).get("period", "5y")
    interval = settings_cfg.get("fetcher", {}).get("interval", "1d")

    raw_prices_dir = BASE_DIR / settings_cfg.get("paths", {}).get("raw_prices_dir", "data/raw/prices")
    raw_indices_dir = BASE_DIR / settings_cfg.get("paths", {}).get("raw_indices_dir", "data/raw/indices")
    raw_macro_dir = BASE_DIR / settings_cfg.get("paths", {}).get("raw_macro_dir", "data/raw/macro")
    processed_dir = BASE_DIR / settings_cfg.get("paths", {}).get("processed_dir", "data/processed")
    metadata_dir = BASE_DIR / settings_cfg.get("paths", {}).get("metadata_dir", "data/metadata")
    reports_dir = BASE_DIR / settings_cfg.get("paths", {}).get("reports_dir", "reports")

    for d in [raw_prices_dir, raw_indices_dir, raw_macro_dir, processed_dir, metadata_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Gather all stock tickers (Nifty 50, Bank, IT, Next 50)
    stocks_dict = tickers_cfg.get("stocks", {})
    all_stock_tickers = []
    for grp_name, ticker_list in stocks_dict.items():
        all_stock_tickers.extend(ticker_list)
    all_stock_tickers = sorted(list(set(all_stock_tickers)))

    indices_dict = tickers_cfg.get("indices", {})
    macro_dict = tickers_cfg.get("macro", {})

    logger.info(f"Targeting {len(all_stock_tickers)} stock tickers across NIFTY 50, Bank, IT, Next 50...")

    # 2. Fetch Raw Data
    stocks_raw_df, excluded_stocks = fetch_multiple_stocks(all_stock_tickers, period=period, interval=interval)
    indices_raw_dfs = fetch_all_indices(indices_dict, period=period, interval=interval)
    macro_raw_dfs = fetch_all_macro(macro_dict, period=period, interval=interval)

    # Save raw datasets
    if not stocks_raw_df.empty:
        stocks_raw_df.to_parquet(raw_prices_dir / "all_stocks_raw.parquet", index=False)

    for idx_name, df_idx in indices_raw_dfs.items():
        df_idx.to_parquet(raw_indices_dir / f"{idx_name}.parquet", index=False)

    for mac_name, df_mac in macro_raw_dfs.items():
        df_mac.to_parquet(raw_macro_dir / f"{mac_name}.parquet", index=False)

    # Save excluded tickers metadata
    excluded_df = pd.DataFrame(excluded_stocks) if excluded_stocks else pd.DataFrame(columns=["ticker", "reason", "rows_fetched"])
    excluded_df.to_csv(reports_dir / "excluded_tickers.csv", index=False)
    excluded_df.to_csv(metadata_dir / "excluded_tickers.csv", index=False)

    # 3. Clean & Align Data
    aligned_df = align_and_clean_dataset(stocks_raw_df, indices_raw_dfs, macro_raw_dfs)
    if aligned_df.empty:
        logger.error("Pipeline failed: Aligned dataset is empty.")
        return

    # 4. Generate 100+ Technical Features
    tech_df = generate_technical_features(aligned_df)

    # 5. Generate Lag Features (Zero Future Data Leakage)
    lag_df = generate_lag_features(tech_df)

    # 6. Detect Market Regimes
    final_df = detect_market_regime(lag_df)

    # Warmup filter (drop early NaNs from 50-day moving average initialization)
    if "sma_50" in final_df.columns:
        final_df = final_df.dropna(subset=["sma_50"]).reset_index(drop=True)

    # 7. Export Final Dataset
    output_parquet_path = BASE_DIR / settings_cfg.get("processed", {}).get("output_parquet", "data/processed/price_features.parquet")
    final_df.to_parquet(output_parquet_path, index=False)
    logger.info(f"SUCCESS! Exported final dataset ({final_df.shape[0]} rows, {final_df.shape[1]} columns) -> {output_parquet_path.relative_to(BASE_DIR)}")

    # 8. Data Quality Report
    date_min = str(final_df["date"].min().date()) if "date" in final_df.columns else "N/A"
    date_max = str(final_df["date"].max().date()) if "date" in final_df.columns else "N/A"
    unique_tickers_cnt = final_df["ticker"].nunique() if "ticker" in final_df.columns else 0
    total_features_cnt = final_df.shape[1]
    total_missing_cells = int(final_df.isna().sum().sum())

    quality_summary = [{
        "total_tickers": unique_tickers_cnt,
        "total_rows": len(final_df),
        "total_columns": total_features_cnt,
        "date_range_start": date_min,
        "date_range_end": date_max,
        "total_missing_values": total_missing_cells,
        "excluded_tickers_count": len(excluded_stocks),
        "output_file": str(output_parquet_path.relative_to(BASE_DIR))
    }]

    quality_report_df = pd.DataFrame(quality_summary)
    quality_report_df.to_csv(reports_dir / "data_quality_report.csv", index=False)
    quality_report_df.to_csv(metadata_dir / "data_quality_report.csv", index=False)
    logger.info(f"Saved Data Quality Report -> {reports_dir / 'data_quality_report.csv'}")

    logger.info("==================================================")
    logger.info("M1 Feature Pipeline Execution Completed Successfully!")
    logger.info("==================================================")

if __name__ == "__main__":
    build_pipeline()
