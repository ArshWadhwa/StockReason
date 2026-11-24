"""
StockReason v0.9.2 — Multi-Horizon Conformal Prediction Interval Calibrator.
=============================================================================
Remediates Item 9:
- Calibrates absolute-price prediction intervals SEPARATELY for every horizon:
  H in {1, 5, 21, 63, 126}.
- Nonconformity scores on volatility-standardized absolute return residuals:
  s_{i, H} = | y_{i, H} - y_hat_{i, H} | / ( sigma_{i, H} + 1e-4 )
  where sigma_{i, H} = vol_60d_i * sqrt(H / 252).
- Dynamic per-stock interval in INR:
  P_hat +- q_{1-alpha, H} * (P_t * sigma_{i, H})
- Evaluates empirical coverage across 50%, 80%, 90%, 95% separately for each horizon.
- Directional probability is strictly OMITTED unless out-of-fold BSS > 0.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class MultiHorizonConformalCalibrator:
    def __init__(
        self,
        horizons: List[int] = [1, 5, 21, 63, 126],
        target_coverages: List[float] = [0.50, 0.80, 0.90, 0.95],
    ):
        self.horizons = list(horizons)
        self.target_coverages = list(target_coverages)
        # Dictionary mapping horizon -> coverage -> quantile multiplier
        self.quantiles: Dict[int, Dict[float, float]] = {h: {} for h in self.horizons}

    def fit_horizon(
        self,
        horizon: int,
        y_true_ret: np.ndarray,
        y_pred_ret: np.ndarray,
        stock_vols_annual: np.ndarray,
    ):
        """Fits conformal quantiles for a specific horizon."""
        holding_vols = np.maximum(stock_vols_annual * np.sqrt(horizon / 252.0), 0.005)
        scores = np.abs(y_true_ret - y_pred_ret) / holding_vols
        scores = scores[~np.isnan(scores)]
        n = len(scores)
        if n == 0:
            for cov in self.target_coverages:
                self.quantiles[horizon][cov] = 1.96
            return

        for cov in self.target_coverages:
            q_level = min(1.0, np.ceil((n + 1) * cov) / n)
            q_val = float(np.quantile(scores, q_level))
            self.quantiles[horizon][cov] = q_val

    def predict_price_interval(
        self,
        pred_return: float,
        stock_vol_annual: float,
        current_price: float,
        horizon: int = 21,
        coverage: float = 0.95,
    ) -> Dict[str, float]:
        """
        Produces dynamically scaled prediction interval in INR for horizon H.
        """
        q_val = self.quantiles.get(horizon, {}).get(coverage, 1.96)
        stock_vol_h = max(stock_vol_annual * np.sqrt(horizon / 252.0), 0.005)
        margin_ret = q_val * stock_vol_h
        
        pred_price = current_price * (1.0 + pred_return)
        margin_inr = current_price * margin_ret
        lower_price = max(0.0, pred_price - margin_inr)
        upper_price = pred_price + margin_inr

        return {
            "horizon_days": horizon,
            "current_price": current_price,
            "pred_return": pred_return,
            "pred_price": pred_price,
            "coverage_nominal": coverage,
            "conformal_multiplier": q_val,
            "margin_inr": margin_inr,
            "lower_price_inr": lower_price,
            "upper_price_inr": upper_price,
        }

    def evaluate_multi_horizon_coverage(
        self,
        val_df: pd.DataFrame,
        preds_by_horizon: Dict[int, np.ndarray],
    ) -> pd.DataFrame:
        """
        Evaluates empirical coverage separately for every horizon across nominal levels.
        """
        rows = []
        for h in self.horizons:
            target_col = f"target_return_{h}d"
            if target_col not in val_df.columns or h not in preds_by_horizon:
                continue

            y_true = val_df[target_col].values
            y_pred = preds_by_horizon[h]
            vols = val_df["vol_60d"].values if "vol_60d" in val_df.columns else np.full(len(val_df), 0.25)
            holding_vols = np.maximum(vols * np.sqrt(h / 252.0), 0.005)

            errors = np.abs(y_true - y_pred)
            valid_m = ~np.isnan(errors)
            errors_clean = errors[valid_m]
            holding_vols_clean = holding_vols[valid_m]
            row = {"horizon": f"{h}d"}

            for cov in self.target_coverages:
                q_val = self.quantiles.get(h, {}).get(cov, 1.96)
                allowed_margin = q_val * holding_vols_clean
                cov_emp = float(np.mean(errors_clean <= allowed_margin) * 100.0) if len(errors_clean) > 0 else 0.0
                row[f"coverage_{int(cov*100)}pct"] = round(cov_emp, 2)
                row[f"multiplier_{int(cov*100)}pct"] = round(q_val, 3)

            rows.append(row)
        return pd.DataFrame(rows)
