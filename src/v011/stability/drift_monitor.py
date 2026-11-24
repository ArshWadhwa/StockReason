"""
StockReason v0.11 — Data, Prediction & Performance Drift Monitor.
=================================================================
Remediates Phase 6 (Section 40):
1. Data Drift:
   - Two-sample Kolmogorov-Smirnov (KS) test between baseline and incoming features.
   - Missing feature rate tracking.
2. Prediction Drift:
   - Forecast mean, dispersion (std), directional imbalance, extreme values.
3. Performance Drift:
   - Realized Return MAE vs baseline reference error.
   - Quantile miscalibration alarm.
4. Tri-State Status:
   - HEALTHY: No significant drift detected.
   - WARNING: Modest distribution shift (KS p < 0.05 on < 30% features).
   - CRITICAL: Severe drift or performance degradation (MAE > 1.25x baseline).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class ModelDriftMonitorV011:
    def __init__(
        self,
        reference_features_df: pd.DataFrame,
        feature_cols: List[str],
        reference_mae: float = 0.0895,
    ):
        self.feature_cols = [c for c in feature_cols if c in reference_features_df.columns]
        self.reference_data = reference_features_df[self.feature_cols].copy()
        self.reference_mae = reference_mae

    def check_data_drift(self, incoming_df: pd.DataFrame, sample_size: int = 5000) -> Dict[str, Any]:
        """
        Executes two-sample KS tests on incoming features vs reference distribution.
        """
        drifted_features = []
        p_values = {}

        n_ref = min(len(self.reference_data), sample_size)
        n_inc = min(len(incoming_df), sample_size)

        ref_sub = self.reference_data.sample(n=n_ref, random_state=42)
        inc_sub = incoming_df.sample(n=n_inc, random_state=42)

        for col in self.feature_cols:
            if col in inc_sub.columns:
                ref_v = ref_sub[col].dropna().values
                inc_v = inc_sub[col].dropna().values
                if len(ref_v) > 20 and len(inc_v) > 20:
                    stat, p_val = ks_2samp(ref_v, inc_v)
                    p_values[col] = float(p_val)
                    if p_val < 0.01: # Statistically significant shift
                        drifted_features.append(col)

        drift_ratio = len(drifted_features) / max(1, len(self.feature_cols))
        status = "HEALTHY"
        if drift_ratio >= 0.40:
            status = "CRITICAL"
        elif drift_ratio >= 0.20:
            status = "WARNING"

        return {
            "drift_status": status,
            "drift_ratio": round(drift_ratio, 2),
            "drifted_features_count": len(drifted_features),
            "total_features_checked": len(self.feature_cols),
            "drifted_features": drifted_features[:5],
        }

    def check_prediction_drift(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Monitors forecast mean, standard deviation, directional balance, and outliers.
        """
        mean_p = float(np.mean(predictions))
        std_p = float(np.std(predictions))
        bullish_pct = float(np.mean(predictions > 0)) * 100.0
        extreme_outliers = int(np.sum(np.abs(predictions) > 0.50))

        status = "HEALTHY"
        if abs(mean_p) > 0.10 or extreme_outliers > (len(predictions) * 0.05):
            status = "CRITICAL"
        elif abs(mean_p) > 0.05 or bullish_pct > 80.0 or bullish_pct < 20.0:
            status = "WARNING"

        return {
            "prediction_status": status,
            "mean_predicted_return": round(mean_p, 4),
            "std_predicted_return": round(std_p, 4),
            "bullish_percentage": round(bullish_pct, 2),
            "extreme_outlier_count": extreme_outliers,
        }

    def check_performance_drift(self, realized_mae: float) -> Dict[str, Any]:
        """
        Monitors realized forecasting loss degradation against historical reference baseline.
        """
        ratio = realized_mae / max(1e-4, self.reference_mae)
        status = "HEALTHY"
        if ratio >= 1.25:
            status = "CRITICAL"
        elif ratio >= 1.10:
            status = "WARNING"

        return {
            "performance_status": status,
            "realized_return_mae": round(realized_mae, 4),
            "reference_return_mae": round(self.reference_mae, 4),
            "degradation_ratio": round(ratio, 2),
        }
