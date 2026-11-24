"""
StockReason v0.10 — Regime-Adaptive Forecasting Ensemble.
==========================================================
Remediates Phase 3 (Section 8):
Blends candidate predictors using causal regime-conditioned weights:
    Final forecast = w_A(Regime) * Drift + w_B(Regime) * Hybrid + w_C(Regime) * LightGBM

Compares:
1. Regime-Adaptive Ensemble (dynamic weights conditioned on trailing regime).
2. Fixed-Weight Control (uniform equal weights across all regimes).
3. Individual Candidate Predictors.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.v092.models.baselines import compute_absolute_price_metrics


# Causal regime weights derived strictly from training development performance
REGIME_WEIGHTS = {
    # Bullish_LowVol: Favor trend/momentum and gradient boosted models
    "Bullish_LowVol": {"hist_mean": 0.20, "hybrid": 0.30, "lgbm": 0.50},
    # Bullish_HighVol: Balance between macro drift and tree ensemble
    "Bullish_HighVol": {"hist_mean": 0.35, "hybrid": 0.35, "lgbm": 0.30},
    # Bearish_Stress: Heavily favor conservative historical drift and CAPM
    "Bearish_Stress": {"hist_mean": 0.60, "hybrid": 0.30, "lgbm": 0.10},
    # Sideways_Neutral: Shrink toward historical mean
    "Sideways_Neutral": {"hist_mean": 0.50, "hybrid": 0.30, "lgbm": 0.20},
}


class RegimeAdaptiveEnsemble:
    def __init__(self, regime_weights: Dict[str, Dict[str, float]] = REGIME_WEIGHTS):
        self.regime_weights = regime_weights

    def predict(
        self,
        df_eval: pd.DataFrame,
        pred_hist_mean: np.ndarray,
        pred_hybrid: np.ndarray,
        pred_lgbm: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generates both Adaptive Ensemble predictions and Fixed-Weight Control predictions.
        """
        n = len(df_eval)
        adaptive_pred = np.zeros(n)
        fixed_pred = (pred_hist_mean + pred_hybrid + pred_lgbm) / 3.0

        regimes = df_eval["market_regime"].values if "market_regime" in df_eval.columns else np.full(n, "Sideways_Neutral")

        for i, reg in enumerate(regimes):
            w = self.regime_weights.get(reg, {"hist_mean": 0.333, "hybrid": 0.333, "lgbm": 0.334})
            adaptive_pred[i] = (
                w["hist_mean"] * pred_hist_mean[i]
                + w["hybrid"] * pred_hybrid[i]
                + w["lgbm"] * pred_lgbm[i]
            )

        close_prices = df_eval["close"].values
        return {
            "adaptive_return": adaptive_pred,
            "adaptive_price": close_prices * (1.0 + adaptive_pred),
            "fixed_control_return": fixed_pred,
            "fixed_control_price": close_prices * (1.0 + fixed_pred),
        }

    def evaluate_comparison(
        self,
        df_eval: pd.DataFrame,
        target_ret: np.ndarray,
        pred_hist_mean: np.ndarray,
        pred_hybrid: np.ndarray,
        pred_lgbm: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compares Adaptive Ensemble vs Fixed-Weight Control vs Individual Candidates.
        """
        preds = self.predict(df_eval, pred_hist_mean, pred_hybrid, pred_lgbm)
        close = df_eval["close"].values

        m_adapt = compute_absolute_price_metrics(target_ret, preds["adaptive_return"], close)
        m_fixed = compute_absolute_price_metrics(target_ret, preds["fixed_control_return"], close)
        m_mean = compute_absolute_price_metrics(target_ret, pred_hist_mean, close)
        m_hyb = compute_absolute_price_metrics(target_ret, pred_hybrid, close)
        m_lgb = compute_absolute_price_metrics(target_ret, pred_lgbm, close)

        records = [
            {"model_variant": "1. Regime-Adaptive Ensemble", **m_adapt},
            {"model_variant": "2. Fixed-Weight Control (Uniform)", **m_fixed},
            {"model_variant": "3. Candidate: Historical Mean", **m_mean},
            {"model_variant": "4. Candidate: Hybrid Decomposition", **m_hyb},
            {"model_variant": "5. Candidate: LightGBM Ensemble", **m_lgb},
        ]
        return pd.DataFrame(records)
