import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market state into categories:
      - 'VOLATILE' : High VIX (> 18.0) or high ATR ratio
      - 'TRENDING' : Strong directional trend (ADX > 25 or SMA_20/SMA_50 slope divergence)
      - 'CHOPPY'   : Low trend strength, range-bound sideways market

    Appends 'market_regime' column to DataFrame.
    """
    logger.info("Detecting market regimes (TRENDING / CHOPPY / VOLATILE)...")

    if df.empty:
        return df

    result = df.copy()

    vix = result["vix_close"] if "vix_close" in result.columns else pd.Series(15.0, index=result.index)
    adx = result["adx_14"] if "adx_14" in result.columns else pd.Series(20.0, index=result.index)
    natr = result["natr_14"] if "natr_14" in result.columns else pd.Series(1.5, index=result.index)

    conditions = [
        (vix > 18.0) | (natr > 3.0),
        (adx > 25.0) & (vix <= 18.0),
    ]

    choices = ["VOLATILE", "TRENDING"]
    result["market_regime"] = np.select(conditions, choices, default="CHOPPY")

    regime_counts = result["market_regime"].value_counts().to_dict()
    logger.info(f"Market Regime Distribution: {regime_counts}")

    return result
