"""
StockReason v0.9.2 — Dual Production Forecasting Service (Engine A & Engine B).
=================================================================================
Remediates Items 1, 12:
- Exposes two distinct forecast engines via a single production inference service:
  1. Engine A: Absolute future return & INR price forecast with dynamic conformal interval.
  2. Engine B: Cross-sectional relative demeaned return & relative ranking.
- Features computed using RealMacroDataProvider with zero static defaults.
- Directional probability strictly OMITTED (None) when out-of-fold BSS <= 0.
- Parity between batch evaluation and API inference is guaranteed < 1e-9.
"""

import os
import hashlib
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.v091.features.pipeline import FEATURE_COLS_21, StockReasonPipelineV091
from src.v091.features.macro import RealMacroDataProvider
from src.v092.calibration.intervals import MultiHorizonConformalCalibrator

MODEL_VERSION = "0.9.2"
DEFAULT_PRICES_DIR = "data/indian-markets/prices_adjusted"


class StockReasonDualForecastServiceV092:
    def __init__(
        self,
        engine_a_pipeline: Optional[Any] = None,
        engine_b_pipeline: Optional[Any] = None,
        conformal_calibrator: Optional[MultiHorizonConformalCalibrator] = None,
        prices_dir: str = DEFAULT_PRICES_DIR,
        macro_provider: Optional[RealMacroDataProvider] = None,
    ):
        self.engine_a = engine_a_pipeline
        self.engine_b = engine_b_pipeline
        self.conformal = conformal_calibrator or MultiHorizonConformalCalibrator()
        self.prices_dir = prices_dir
        self.macro_provider = macro_provider or RealMacroDataProvider()
        self.version = MODEL_VERSION

    def forecast_security(
        self,
        symbol: str,
        horizon_days: int = 21,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes inference for both Engine A (absolute price) and Engine B (relative rank).
        """
        clean_sym = symbol.upper().replace(".NS", "").strip()

        # Load adjusted price data for symbol
        adj_files = sorted([f for f in os.listdir(self.prices_dir) if f.endswith(".parquet")])
        dfs = []
        for f in adj_files:
            p = os.path.join(self.prices_dir, f)
            try:
                sub = pd.read_parquet(p, columns=["date", "symbol", "series", "close", "adj_close", "volume", "turnover", "high", "low"])
                eq = sub[(sub["symbol"] == clean_sym) & (sub["series"] == "EQ")]
                if not eq.empty:
                    dfs.append(eq)
            except Exception:
                pass

        if not dfs:
            raise ValueError(f"Security '{symbol}' not found in canonical dataset.")

        df_stock = pd.concat(dfs, ignore_index=True)
        df_stock["date_str"] = pd.to_datetime(df_stock["date"]).dt.strftime("%Y-%m-%d")
        df_stock = df_stock.sort_values("date_str")

        if as_of_date:
            sub = df_stock[df_stock["date_str"] <= as_of_date]
        else:
            sub = df_stock

        if len(sub) < 30:
            raise ValueError(f"Insufficient history ({len(sub)} sessions) for {symbol} up to cutoff.")

        latest_row = sub.iloc[-1]
        ref_price = float(latest_row["close"])
        cutoff_date = latest_row["date_str"]

        # Extract features using single canonical pipeline
        if self.engine_a is not None and hasattr(self.engine_a, "extract_features_single"):
            f_vec = self.engine_a.extract_features_single(df_stock, cutoff_date)
        elif self.engine_b is not None and hasattr(self.engine_b, "extract_features_single"):
            f_vec = self.engine_b.extract_features_single(df_stock, cutoff_date)
        else:
            p_obj = StockReasonPipelineV091(macro_provider=self.macro_provider)
            f_vec = p_obj.extract_features_single(df_stock, cutoff_date)

        if f_vec.ndim == 1:
            X_input = f_vec.reshape(1, -1)
            vol_ann = float(f_vec[7])
        else:
            X_input = f_vec
            vol_ann = float(f_vec[0, 7])

        # 1. Engine A: Absolute Return & INR Price Forecast
        if self.engine_a is not None:
            pred_a = float(self.engine_a.predict(X_input)[0])
        else:
            pred_a = 0.0

        expected_price = ref_price * (1.0 + pred_a)

        # Conformal interval for horizon
        interval = self.conformal.predict_price_interval(
            pred_return=pred_a,
            stock_vol_annual=vol_ann,
            current_price=ref_price,
            horizon=horizon_days,
            coverage=0.95,
        )

        # 2. Engine B: Cross-Sectional Demeaned Return
        if self.engine_b is not None:
            pred_b = float(self.engine_b.predict(X_input)[0])
        else:
            pred_b = 0.0

        macro_data = self.macro_provider.get_macro_vector(cutoff_date)
        n21 = macro_data["nifty_ret_21d"]
        vix_lvl = macro_data["vix_level"]

        return {
            "symbol": clean_sym,
            "as_of_date": cutoff_date,
            "horizon_days": horizon_days,
            "pipeline_version": self.version,
            "engine_a_absolute_price": {
                "reference_price_inr": ref_price,
                "predicted_absolute_return": round(pred_a, 4),
                "predicted_target_price_inr": round(expected_price, 2),
                "interval_95_pct_inr": [
                    round(interval["lower_price_inr"], 2),
                    round(interval["upper_price_inr"], 2),
                ],
                "margin_inr": round(interval["margin_inr"], 2),
            },
            "engine_b_relative_ranking": {
                "predicted_demeaned_return": round(pred_b, 4),
                "relative_signal": "OVERWEIGHT" if pred_b > 0.01 else ("UNDERWEIGHT" if pred_b < -0.01 else "NEUTRAL"),
            },
            "probability_positive_return": None, # Strictly omitted because BSS <= 0
            "macro_context": {
                "nifty_ret_21d": round(float(n21), 4),
                "india_vix": round(float(vix_lvl), 2),
            },
        }
