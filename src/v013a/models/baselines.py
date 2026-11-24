"""StockReason v0.13A — Statistical Baselines Suite Across All 5 Horizons.

Section D Requirements:
1. Re-runs genuine statistical baselines across all 5 horizons (1, 5, 21, 63, 126):
   - No-change Random Walk (pred_return = 0.0)
   - Historical Mean Return (fitted strictly on preceding training data)
   - Historical Median Return (fitted strictly on preceding training data)
   - Hierarchical Shrinkage Baseline (shrunk toward sector/market mean)
2. Strictly identical evaluation cohorts.
3. Zero-lookahead: fitted strictly on pre-2023 training partition with horizon-specific label purging.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("v013a_baselines")


class StatisticalBaselinesSuite:
    """
    Computes statistical benchmark forecasts across all horizons
    fitted strictly on authorized preceding training data.
    """

    def __init__(self, horizons: List[int] = [1, 5, 21, 63, 126]):
        self.horizons = horizons
        self.historical_means: Dict[int, float] = {}
        self.historical_medians: Dict[int, float] = {}
        self.stock_medians: Dict[int, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame):
        """Fits historical return distributions strictly on training partition."""
        for h in self.horizons:
            ret_col = f"target_return_{h}d"
            if ret_col in train_df.columns:
                valid_ret = train_df[ret_col].dropna().values
                if len(valid_ret) > 0:
                    self.historical_means[h] = float(np.mean(valid_ret))
                    self.historical_medians[h] = float(np.median(valid_ret))
                    
                    # Stock-level medians
                    stock_grp = train_df.groupby("symbol")[ret_col].median().to_dict()
                    self.stock_medians[h] = stock_grp
                    logger.info("Horizon %3dd Training Stats: Mean=%.6f, Median=%.6f (across %d obs)",
                                h, self.historical_means[h], self.historical_medians[h], len(valid_ret))

    def predict(
        self,
        eval_df: pd.DataFrame,
        horizon: int = 21,
        alpha_shrinkage: float = 0.50,
    ) -> pd.DataFrame:
        """
        Generates predictions for all statistical baselines on evaluation partition.
        Returns DataFrame with pred_ret_* and pred_price_* columns.
        """
        out = eval_df.copy()
        close = out["close"].values
        symbols = out["symbol"].values

        # 1. No-Change Random Walk
        out[f"pred_ret_rw_{horizon}d"] = 0.0
        out[f"pred_price_rw_{horizon}d"] = close * 1.0

        # 2. Historical Mean Return
        mean_h = self.historical_means.get(horizon, 0.0)
        out[f"pred_ret_mean_{horizon}d"] = mean_h
        out[f"pred_price_mean_{horizon}d"] = close * (1.0 + mean_h)

        # 3. Historical Median Return
        med_h = self.historical_medians.get(horizon, 0.0)
        out[f"pred_ret_median_{horizon}d"] = med_h
        out[f"pred_price_median_{horizon}d"] = close * (1.0 + med_h)

        # 4. Hierarchical Shrinkage: stock median shrunk toward market median
        stock_med_map = self.stock_medians.get(horizon, {})
        stock_meds = np.array([stock_med_map.get(s, med_h) for s in symbols], dtype=np.float64)
        stock_meds = np.nan_to_num(stock_meds, nan=med_h)
        shrunk_ret = (1.0 - alpha_shrinkage) * stock_meds + alpha_shrinkage * med_h
        out[f"pred_ret_shrinkage_{horizon}d"] = shrunk_ret
        out[f"pred_price_shrinkage_{horizon}d"] = close * (1.0 + shrunk_ret)

        return out
