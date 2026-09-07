import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_lag_features(df: pd.DataFrame, lag_periods: list = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
    """
    Generate historical lag features for returns, volume, and core indicators.
    CRITICAL: Uses shift(k) with k >= 1 to strictly prevent future data leakage.
    """
    logger.info(f"Generating lag features for periods: {lag_periods} (zero future leakage)...")

    if df.empty:
        return df

    result_df = df.copy()
    target_cols = [c for c in ["return_1d", "volume", "rsi_14", "macd_line_12_26", "volatility_20d", "atr_14"] if c in result_df.columns]

    def _compute_lags(group_df: pd.DataFrame) -> pd.DataFrame:
        g = group_df.copy()
        for col in target_cols:
            for lag in lag_periods:
                g[f"{col}_lag_{lag}"] = g[col].shift(lag)
        return g

    if "ticker" in result_df.columns:
        dfs = []
        for ticker, group_df in result_df.groupby("ticker"):
            g = _compute_lags(group_df)
            g["ticker"] = ticker
            dfs.append(g)
        result_df = pd.concat(dfs, ignore_index=True)
    else:
        result_df = _compute_lags(result_df)

    new_lag_cols = [c for c in result_df.columns if "_lag_" in c]
    logger.info(f"Generated {len(new_lag_cols)} lag features.")
    return result_df
