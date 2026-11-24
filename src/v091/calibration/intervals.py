"""
StockReason v0.9.1 — Conformal Prediction Intervals on Volatility-Standardized Residuals.
========================================================================================
Remediates the Static Uncertainty Defect (Task 7):
In v0.9:
- `forecast_service.py` gave EVERY stock the exact same +-15% interval:
  interval = pred_price * (1.0 +- 1.96 * 0.076), regardless of whether the stock
  was an ultra-low-volatility utility or a high-volatility penny stock.
- Furthermore, it scaled intervals across horizons with sqrt(h / 21).

In v0.9.1:
- Conformal calibration is fitted on volatility-standardized residuals:
  nonconformity score s_i = | y_i - y_hat_i | / ( sigma_i + 1e-4 )
  where sigma_i is the stock's trailing 60-day annualized volatility (scaled to 21-day holding).
- At nominal coverage level 1 - alpha (50%, 80%, 90%, 95%), conformal multiplier q_{1-alpha}
  is the (1 - alpha)(1 + 1/n) empirical quantile of calibration scores.
- Stock-specific interval:
  [ y_hat_j - q_{1-alpha} * sigma_j, y_hat_j + q_{1-alpha} * sigma_j ]
- Each stock's width dynamically reflects its own intrinsic volatility.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class VolatilityStandardizedConformalCalibrator:
    def __init__(self, target_coverages: List[float] = [0.50, 0.80, 0.90, 0.95]):
        self.target_coverages = target_coverages
        self.quantiles: Dict[float, float] = {}

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, volatilities: np.ndarray):
        """
        Fits conformal nonconformity quantiles on out-of-fold calibration data.
        """
        # Convert annualized volatility to 21-day volatility: sigma_21d = sigma_ann * sqrt(21 / 252)
        vols_holding = np.maximum(volatilities * np.sqrt(21.0 / 252.0), 0.01)
        scores = np.abs(y_true - y_pred) / vols_holding
        scores = scores[~np.isnan(scores)]
        
        n = len(scores)
        for cov in self.target_coverages:
            # Finite-sample adjusted conformal quantile
            q_level = min(1.0, np.ceil((n + 1) * cov) / n)
            q_val = float(np.quantile(scores, q_level))
            self.quantiles[cov] = q_val

    def predict_interval(
        self,
        pred_return: float,
        stock_vol_annual: float,
        ref_price: float,
        coverage: float = 0.95,
    ) -> Dict[str, float]:
        """
        Generates dynamically scaled prediction interval for an individual stock.
        """
        q_val = self.quantiles.get(coverage, 1.96)
        stock_vol_21d = max(stock_vol_annual * np.sqrt(21.0 / 252.0), 0.01)
        margin = q_val * stock_vol_21d
        
        lower_ret = pred_return - margin
        upper_ret = pred_return + margin
        lower_price = ref_price * (1.0 + lower_ret)
        upper_price = ref_price * (1.0 + upper_ret)
        
        return {
            "pred_return": pred_return,
            "margin_return": margin,
            "lower_return": lower_ret,
            "upper_return": upper_ret,
            "reference_price": ref_price,
            "lower_price": lower_price,
            "upper_price": upper_price,
            "stock_vol_21d": stock_vol_21d,
            "conformal_multiplier": q_val
        }

    def evaluate_empirical_coverage(
        self,
        df_eval: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
        vol_col: str = "vol_60d",
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage at 50, 80, 90, 95% overall, per year, and by volatility tercile.
        """
        df = df_eval.dropna(subset=[y_true_col, y_pred_col, vol_col]).copy()
        y_true = df[y_true_col].values
        y_pred = df[y_pred_col].values
        vols_holding = np.maximum(df[vol_col].values * np.sqrt(21.0 / 252.0), 0.01)
        
        results = {}
        for cov in self.target_coverages:
            q_val = self.quantiles.get(cov, 1.96)
            margins = q_val * vols_holding
            is_covered = (y_true >= y_pred - margins) & (y_true <= y_pred + margins)
            
            df[f"covered_{int(cov*100)}"] = is_covered
            results[f"overall_coverage_{int(cov*100)}"] = float(np.mean(is_covered))
            
        # By Year
        if "date_str" in df.columns:
            df["year"] = pd.to_datetime(df["date_str"]).dt.year
            by_year = {}
            for y, grp in df.groupby("year"):
                by_year[int(y)] = {int(cov*100): float(np.mean(grp[f"covered_{int(cov*100)}"])) for cov in self.target_coverages}
            results["coverage_by_year"] = by_year
            
        # By Volatility Tercile
        df["vol_tercile"] = pd.qcut(df[vol_col], q=3, labels=["Low", "Medium", "High"])
        by_tercile = {}
        for t, grp in df.groupby("vol_tercile", observed=False):
            by_tercile[str(t)] = {int(cov*100): float(np.mean(grp[f"covered_{int(cov*100)}"])) for cov in self.target_coverages}
        results["coverage_by_vol_tercile"] = by_tercile
        
        return results
