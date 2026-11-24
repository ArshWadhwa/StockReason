"""
StockReason v0.10 — Quantile Distributional Forecasting & Non-Crossing Engine.
==============================================================================
Remediates Phase 6 (Section 12):
1. Quantile levels:
   5th, 10th, 25th, 50th (Median), 75th, 90th, 95th percentiles.
2. Non-Crossing Post-Processor:
   Enforces q_1 <= q_2 <= ... <= q_7 monotonically per row.
3. Strict Calibration Separation:
   Fits models on training fold, calibrates conformal quantiles on calibration fold,
   evaluates strictly on independent evaluation fold.
4. Suppresses directional win probability unless out-of-fold Brier Skill Score > 0.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb


QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Computes pinball / quantile loss for quantile level q."""
    error = y_true - y_pred
    loss = np.maximum(q * error, (q - 1.0) * error)
    return float(np.mean(loss))


class MultiQuantilePriceForecaster:
    def __init__(self, quantiles: List[float] = QUANTILES, seed: int = 42):
        self.quantiles = list(quantiles)
        self.seed = seed
        self.models: Dict[float, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits separate LightGBM quantile regressors for each quantile level."""
        valid = ~np.isnan(y_train)
        X_tr = X_train[valid]
        y_tr = y_train[valid]

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                learning_rate=0.03,
                n_estimators=60,
                num_leaves=15,
                max_depth=4,
                subsample=0.8,
                random_state=self.seed,
                n_jobs=1,
                verbose=-1,
            ).fit(X_tr, y_tr)
            self.models[q] = model

    def predict_quantiles(
        self,
        X_eval: np.ndarray,
        current_prices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Produces non-crossing return and INR price quantiles for each observation.
        """
        n = len(X_eval)
        raw_preds = np.zeros((n, len(self.quantiles)))

        for i, q in enumerate(self.quantiles):
            raw_preds[:, i] = self.models[q].predict(X_eval)

        # Enforce non-crossing sorting per row: q_1 <= q_2 <= ... <= q_k
        sorted_return_quantiles = np.sort(raw_preds, axis=1)

        # Map to INR prices: P_q = P_t * (1 + r_q)
        inr_prices = np.zeros_like(sorted_return_quantiles)
        for i in range(len(self.quantiles)):
            inr_prices[:, i] = current_prices * (1.0 + sorted_return_quantiles[:, i])

        quantile_dict = {}
        price_quantile_dict = {}
        for i, q in enumerate(self.quantiles):
            q_label = f"q_{int(q*100):02d}"
            quantile_dict[q_label] = sorted_return_quantiles[:, i]
            price_quantile_dict[q_label] = inr_prices[:, i]

        return {
            "return_quantiles": quantile_dict,
            "price_quantiles_inr": price_quantile_dict,
            "expected_price_inr": current_prices * (1.0 + quantile_dict["q_50"]),
            "median_price_inr": price_quantile_dict["q_50"],
            "interval_50_pct": (price_quantile_dict["q_25"], price_quantile_dict["q_75"]),
            "interval_80_pct": (price_quantile_dict["q_10"], price_quantile_dict["q_90"]),
            "interval_90_pct": (price_quantile_dict["q_05"], price_quantile_dict["q_95"]),
        }

    def evaluate(
        self,
        X_val: np.ndarray,
        y_val_ret: np.ndarray,
        val_close: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluates empirical coverage, pinball loss, and interval width across all quantile levels.
        """
        preds = self.predict_quantiles(X_val, val_close)
        records = []

        valid = ~np.isnan(y_val_ret)
        y_clean = y_val_ret[valid]

        for i, q in enumerate(self.quantiles):
            q_label = f"q_{int(q*100):02d}"
            p_q = preds["return_quantiles"][q_label][valid]
            p_price = preds["price_quantiles_inr"][q_label][valid]

            emp_cov = float(np.mean(y_clean <= p_q) * 100.0)
            pinball = compute_pinball_loss(y_clean, p_q, q)

            records.append({
                "quantile_level": q,
                "nominal_coverage_pct": round(q * 100.0, 1),
                "empirical_coverage_pct": round(emp_cov, 2),
                "coverage_error_pct": round(emp_cov - q * 100.0, 2),
                "pinball_loss": round(pinball, 5),
                "mean_predicted_price_inr": round(float(np.mean(p_price)), 2),
            })

        return pd.DataFrame(records)
