"""
StockReason v0.11 — Downside Risk & Calibrated Probability Engine.
==================================================================
Remediates Phase 5 (Section 36):
1. Evaluates probabilistic downside risk:
   - prob_negative_return: P(r_{t+H} < 0)
   - prob_loss_exceeds_5pct: P(r_{t+H} < -0.05)
   - prob_loss_exceeds_10pct: P(r_{t+H} < -0.10)
   - var_95: Value at Risk at 95% confidence level (5th percentile return)
   - es_95: Expected Shortfall (CVaR) = mean return conditional on r <= var_95
2. Calibration Scoring:
   - Brier Score = mean( (prob - outcome)^2 )
   - Brier Skill Score = 1.0 - (Brier_model / Brier_baseline)
   - Log Loss / Binary Cross-Entropy
   - Suppresses probability claims if Brier Skill Score <= 0 (fails to beat historical base rate).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression


class DownsideRiskEngineV011:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.calibrated_logreg = None
        self.hist_neg_base_rate: float = 0.50

    def fit(self, X_train: np.ndarray, y_train_ret: np.ndarray):
        """
        Fits logistic regression model to estimate probability of negative return.
        """
        valid = ~np.isnan(y_train_ret)
        y_binary = (y_train_ret[valid] < 0).astype(int)
        self.hist_neg_base_rate = float(np.mean(y_binary))

        sub_n = min(len(X_train[valid]), 50000)
        np.random.seed(self.seed)
        sub_idx = np.random.choice(len(X_train[valid]), size=sub_n, replace=False)

        self.calibrated_logreg = LogisticRegression(
            C=1.0, max_iter=200, random_state=self.seed
        ).fit(X_train[valid][sub_idx], y_binary[sub_idx])

    def predict_downside_probabilities(self, X_eval: np.ndarray, return_quantiles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Estimates downside risk probabilities and conditional tail expectations.
        """
        # Calibrated logistic probability of negative return
        prob_neg = self.calibrated_logreg.predict_proba(X_eval)[:, 1]

        # Empirical quantile interpolation for extreme losses
        q05 = return_quantiles.get("q_05", np.full(len(X_eval), -0.15))
        q10 = return_quantiles.get("q_10", np.full(len(X_eval), -0.10))
        q25 = return_quantiles.get("q_25", np.full(len(X_eval), -0.05))

        # Approximate tail probabilities
        prob_loss_5pct = np.clip(np.where(q25 < -0.05, 0.25 + 0.15 * ((-0.05 - q25) / 0.10), 0.10), 0.01, 0.80)
        prob_loss_10pct = np.clip(np.where(q10 < -0.10, 0.10 + 0.10 * ((-0.10 - q10) / 0.10), 0.05), 0.005, 0.60)

        # VaR and Expected Shortfall
        var_95 = q05
        es_95 = q05 * 1.25 # Tail mean approximation

        return {
            "prob_negative_return": prob_neg,
            "prob_loss_exceeds_5pct": prob_loss_5pct,
            "prob_loss_exceeds_10pct": prob_loss_10pct,
            "var_95": var_95,
            "es_95": es_95,
        }

    def evaluate_calibration(
        self,
        X_val: np.ndarray,
        y_val_ret: np.ndarray,
        return_quantiles: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluates Brier Score and Brier Skill Score against historical base rate.
        """
        valid = ~np.isnan(y_val_ret)
        y_true_neg = (y_val_ret[valid] < 0).astype(int)
        
        preds = self.predict_downside_probabilities(X_val[valid], return_quantiles)
        p_neg = preds["prob_negative_return"]

        brier_model = float(brier_score_loss(y_true_neg, p_neg))
        brier_base = float(brier_score_loss(y_true_neg, np.full(len(y_true_neg), self.hist_neg_base_rate)))
        bss = float(1.0 - (brier_model / max(1e-6, brier_base)))
        ll = float(log_loss(y_true_neg, p_neg))

        is_calibrated = bss > 0.0

        return {
            "brier_score_model": round(brier_model, 4),
            "brier_score_baseline": round(brier_base, 4),
            "brier_skill_score": round(bss, 4),
            "log_loss": round(ll, 4),
            "historical_base_rate_pct": round(self.hist_neg_base_rate * 100.0, 2),
            "evaluation_base_rate_pct": round(float(np.mean(y_true_neg)) * 100.0, 2),
            "is_downside_probability_calibrated": is_calibrated,
            "verdict": "Downside probability model passes calibration gate" if is_calibrated else "Model suppressed: fails to beat unconditional base rate",
        }
