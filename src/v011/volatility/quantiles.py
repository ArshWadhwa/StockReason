"""
StockReason v0.11 — Quantile Distributional Forecasting & Non-Crossing Engine.
==============================================================================
Remediates Phase 5 (Section 34):
1. Quantile levels:
   5th, 10th, 25th, 50th (Median), 75th, 90th, 95th percentiles.
2. Monotonic Non-Crossing Enforcement:
   q_05 <= q_10 <= q_25 <= q_50 <= q_75 <= q_90 <= q_95 per observation.
3. Positive Price Floor Guarantee:
   P_q = max(0.01, P_t * (1 + r_q)) for all quantiles.
4. Separate Chronological Calibration:
   Fits models on training fold, calibrates conformal offsets on calibration fold,
   evaluates strictly on independent evaluation fold.
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


def compute_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Computes Winkler Interval Score for central (1 - alpha) prediction interval:
    IS = (upper - lower) + (2/alpha) * (lower - y) * 1(y < lower) + (2/alpha) * (y - upper) * 1(y > upper)
    """
    width = upper - lower
    under = np.maximum(lower - y_true, 0.0)
    over = np.maximum(y_true - upper, 0.0)
    score = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    return float(np.mean(score))


class MultiQuantilePriceForecasterV011:
    def __init__(self, quantiles: List[float] = QUANTILES, seed: int = 42):
        self.quantiles = list(quantiles)
        self.seed = seed
        self.models: Dict[float, Any] = {}
        self.conformal_offsets: Dict[float, float] = {q: 0.0 for q in self.quantiles}

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
                n_estimators=70,
                num_leaves=25,
                max_depth=5,
                subsample=0.8,
                random_state=self.seed,
                n_jobs=1,
                verbose=-1,
            ).fit(X_tr, y_tr)
            self.models[q] = model

    def calibrate_conformal_offsets(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrates residual empirical coverage on independent calibration fold.
        """
        valid = ~np.isnan(y_cal)
        X_c = X_cal[valid]
        y_c = y_cal[valid]

        for q in self.quantiles:
            raw_p = self.models[q].predict(X_c)
            residuals = y_c - raw_p
            # Conformal shift to match nominal coverage
            self.conformal_offsets[q] = float(np.quantile(residuals, q))

    def predict_quantiles(
        self,
        X_eval: np.ndarray,
        current_prices: np.ndarray,
        apply_conformal: bool = True,
    ) -> Dict[str, Any]:
        """
        Produces non-crossing return and INR price quantiles for each observation,
        enforcing positive price floors.
        """
        n = len(X_eval)
        raw_preds = np.zeros((n, len(self.quantiles)))

        for i, q in enumerate(self.quantiles):
            p = self.models[q].predict(X_eval)
            if apply_conformal:
                p = p + self.conformal_offsets[q]
            raw_preds[:, i] = p

        # Enforce non-crossing sorting per row: q_1 <= q_2 <= ... <= q_k
        sorted_return_quantiles = np.sort(raw_preds, axis=1)

        # Map to INR prices: P_q = max(0.01, P_t * (1 + r_q))
        inr_prices = np.zeros_like(sorted_return_quantiles)
        for i in range(len(self.quantiles)):
            inr_prices[:, i] = np.maximum(0.01, current_prices * (1.0 + sorted_return_quantiles[:, i]))

        quantile_dict = {}
        price_quantile_dict = {}
        for i, q in enumerate(self.quantiles):
            q_label = f"q_{int(q*100):02d}"
            quantile_dict[q_label] = sorted_return_quantiles[:, i]
            price_quantile_dict[q_label] = inr_prices[:, i]

        return {
            "return_quantiles": quantile_dict,
            "price_quantiles_inr": price_quantile_dict,
            "expected_price_inr": np.maximum(0.01, current_prices * (1.0 + quantile_dict["q_50"])),
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
        Evaluates empirical coverage, pinball loss, and interval scores.
        """
        valid = ~np.isnan(y_val_ret)
        X_v = X_val[valid]
        y_v = y_val_ret[valid]
        close_v = val_close[valid]

        preds = self.predict_quantiles(X_v, close_v, apply_conformal=True)
        ret_q = preds["return_quantiles"]
        price_q = preds["price_quantiles_inr"]

        results = []
        for q in self.quantiles:
            q_label = f"q_{int(q*100):02d}"
            pred_col = ret_q[q_label]
            emp_cov = float(np.mean(y_v <= pred_col)) * 100.0
            p_loss = compute_pinball_loss(y_v, pred_col, q)

            results.append({
                "quantile_level": q_label,
                "nominal_coverage_pct": round(q * 100.0, 1),
                "empirical_coverage_pct": round(emp_cov, 2),
                "coverage_error_pct": round(emp_cov - q * 100.0, 2),
                "pinball_loss": round(p_loss, 4),
            })

        # Central interval scores
        # 80% interval (q_10 to q_90)
        is_80 = compute_interval_score(y_v, ret_q["q_10"], ret_q["q_90"], alpha=0.20)
        # 90% interval (q_05 to q_95)
        is_90 = compute_interval_score(y_v, ret_q["q_05"], ret_q["q_95"], alpha=0.10)

        df_out = pd.DataFrame(results)
        df_out["interval_score_80_pct"] = round(is_80, 4)
        df_out["interval_score_90_pct"] = round(is_90, 4)
        return df_out
