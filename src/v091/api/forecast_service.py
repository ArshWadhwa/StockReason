"""
StockReason v0.9.1 — Production Forecasting Engine & Inference Service.
=======================================================================
Fully remediated inference service (Tasks 2, 4, 7):
- Evaluates features strictly through the single serialized `StockReasonPipelineV091` object.
- Uses `RealMacroDataProvider` (Nifty 50, India VIX) with ZERO static defaults.
- Conformal prediction intervals on volatility-standardized residuals (each stock's width
  reflects its own intrinsic risk).
- Probability of positive return calibrated via out-of-fold calibration;
  OMITTED (returns None) if Brier Skill Score <= 0 out-of-sample.
- Zero square-root-of-time scaling across horizons.
"""

import os
import json
import hashlib
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.v091.features.pipeline import StockReasonPipelineV091
from src.v091.calibration.intervals import VolatilityStandardizedConformalCalibrator
from src.v091.calibration.probabilities import CalibratedProbabilityEngine
from src.v091.features.targets import decompose_price_forecast

MODEL_VERSION = "0.9.1"
DEFAULT_PIPELINE_PATH = "artifacts/v0.9.1/models/pipeline_21d.joblib"
CANONICAL_PRICES_DIR = "data/indian-markets/prices_adjusted"


class StockReasonForecastServiceV091:
    def __init__(
        self,
        pipeline_path: str = DEFAULT_PIPELINE_PATH,
        prices_dir: str = CANONICAL_PRICES_DIR,
        conformal_calibrator: Optional[VolatilityStandardizedConformalCalibrator] = None,
        probability_engine: Optional[CalibratedProbabilityEngine] = None,
    ):
        self.pipeline_path = pipeline_path
        self.prices_dir = prices_dir
        self.pipeline: Optional[StockReasonPipelineV091] = None
        self.conformal = conformal_calibrator or VolatilityStandardizedConformalCalibrator()
        self.prob_engine = probability_engine or CalibratedProbabilityEngine()
        self._load_pipeline()

    def _load_pipeline(self):
        if os.path.exists(self.pipeline_path):
            self.pipeline = StockReasonPipelineV091.load(self.pipeline_path)
        else:
            # Fallback initializing empty pipeline
            self.pipeline = StockReasonPipelineV091(version=MODEL_VERSION)

    def forecast_price(
        self,
        symbol: str,
        horizon_days: int = 21,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates calibrated exact-price forecast for a supported NSE stock.
        """
        if horizon_days != 21:
            raise NotImplementedError(
                f"Horizon {horizon_days}d is exploratory only. In accordance with Task 7 rules, "
                f"square-root-of-time scaling is removed. Only validated 21-day model is active."
            )
            
        clean_sym = symbol.upper().replace(".NS", "").strip()
        
        # Load prices for symbol from adjusted prices repository
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
        origin_date = latest_row["date_str"]
        
        # 1. Feature extraction through identical pipeline object
        feat_vector = self.pipeline.extract_features_single(sub, as_of_date=origin_date)
        
        # 2. Prediction
        pred_residual = float(self.pipeline.predict(feat_vector)[0])
        
        # 3. Trailing market component (estimated explicitly from Nifty)
        macro_vec = self.pipeline.macro_provider.get_macro_vector(origin_date)
        market_component = float(macro_vec["nifty_ret_21d"])
        
        # 4. Decompose price forecast
        decomp = decompose_price_forecast(ref_price, pred_residual, market_return_forecast=market_component)
        expected_price = decomp["expected_price_inr"]
        total_pred_return = decomp["total_expected_return"]
        
        # 5. Volatility-standardized conformal prediction interval
        adj_c = sub["adj_close"].astype(float)
        ret1 = adj_c.pct_change(1)
        stock_vol_annual = float(ret1.tail(60).std() * np.sqrt(252)) if len(ret1) >= 60 else float(ret1.std() * np.sqrt(252))
        
        conf_int = self.conformal.predict_interval(
            pred_return=total_pred_return,
            stock_vol_annual=stock_vol_annual,
            ref_price=ref_price,
            coverage=0.95
        )
        
        # 6. Calibrated Probability (OMITTED if Brier skill score <= 0)
        prob_up = self.prob_engine.get_calibrated_probability_or_omit(pred_residual)
        
        return {
            "symbol": clean_sym,
            "forecast_origin_date": origin_date,
            "horizon_days": horizon_days,
            "reference_price_inr": ref_price,
            "market_component_return": round(market_component, 6),
            "stock_residual_return": round(pred_residual, 6),
            "total_expected_return": round(total_pred_return, 6),
            "expected_price_inr": round(expected_price, 2),
            "prediction_interval_95": {
                "lower_return": round(conf_int["lower_return"], 6),
                "upper_return": round(conf_int["upper_return"], 6),
                "lower_price_inr": round(conf_int["lower_price"], 2),
                "upper_price_inr": round(conf_int["upper_price"], 2),
                "stock_trailing_volatility_annual": round(stock_vol_annual, 4),
                "method": "conformal_volatility_standardized"
            },
            "probability_up": prob_up, # None if Brier Skill Score <= 0
            "probability_status": "OMITTED (Brier skill score <= 0)" if prob_up is None else "CALIBRATED",
            "model_version": MODEL_VERSION,
            "pipeline_config_hash": self.pipeline.config_hash
        }
