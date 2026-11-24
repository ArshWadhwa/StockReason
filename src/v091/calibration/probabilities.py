"""
StockReason v0.9.1 — Out-of-Fold Probability Calibration & Brier Skill Score.
=============================================================================
Remediates the Naive Normal CDF Probability Defect (Task 7):
In v0.9:
- `forecast_service.py` evaluated probability of positive return using:
  P(up) = Phi( pred / 0.076 )
  which had negative Brier skill score and severe calibration distortion.

In v0.9.1:
- Fits Platt scaling (Logistic Regression) or Isotonic Calibration strictly on
  OUT-OF-FOLD validation predictions (never on test period).
- Evaluates Brier Score against base rate Brier Score:
  BS = (1 / N) * sum (p_i - y_i)^2
  BS_ref = y_bar * (1 - y_bar)
  Brier Skill Score (BSS) = 1 - (BS / BS_ref)
- Computes Expected Calibration Error (ECE) and 10-bin reliability table.
- Mechanical Gate:
  IF Brier Skill Score <= 0 -> OMIT probability from API and UI (returns None).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class CalibratedProbabilityEngine:
    def __init__(self, method: str = "platt"):
        self.method = method
        self.calibrator = None
        self.base_rate: float = 0.5376
        self.brier_skill_score: float = -1.0
        self.is_valid: bool = False

    def fit(self, y_true_binary: np.ndarray, pred_scores: np.ndarray):
        """
        Fits calibration model on out-of-fold validation set.
        y_true_binary: 1 if positive return, 0 otherwise.
        pred_scores: raw continuous predictions or model scores.
        """
        valid_mask = (~np.isnan(y_true_binary)) & (~np.isnan(pred_scores))
        y = y_true_binary[valid_mask].astype(int)
        scores = pred_scores[valid_mask].reshape(-1, 1)
        
        self.base_rate = float(np.mean(y))
        
        if len(y) < 50 or np.var(scores) < 1e-12:
            self.is_valid = False
            return
            
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(scores.ravel(), y)
        else: # Platt scaling
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(scores, y)
            
        # Evaluate Brier Skill Score on calibration cohort
        probs = self.predict_proba(scores.ravel())
        bs = float(np.mean((probs - y) ** 2))
        bs_ref = float(self.base_rate * (1.0 - self.base_rate))
        self.brier_skill_score = float(1.0 - (bs / (bs_ref + 1e-9)))
        
        # Mechanical Gate: Valid only if Brier Skill Score > 0
        self.is_valid = (self.brier_skill_score > 0.0)

    def predict_proba(self, pred_scores: np.ndarray) -> np.ndarray:
        """Computes calibrated probabilities."""
        if self.calibrator is None:
            return np.full_like(pred_scores, self.base_rate, dtype=float)
            
        scores = np.asarray(pred_scores).reshape(-1, 1)
        if hasattr(self.calibrator, "predict_proba"):
            probs = self.calibrator.predict_proba(scores)[:, 1]
        else:
            probs = self.calibrator.predict(scores.ravel())
            
        return np.clip(probs, 0.01, 0.99)

    def get_calibrated_probability_or_omit(self, pred_score: float) -> Optional[float]:
        """
        Returns calibrated probability if BSS > 0; otherwise OMITS (returns None).
        Mandatory gate for API and UI.
        """
        if not self.is_valid or self.brier_skill_score <= 0.0:
            return None
        prob = float(self.predict_proba(np.array([pred_score]))[0])
        return round(prob, 4)

    def evaluate_reliability(
        self,
        y_true_binary: np.ndarray,
        pred_probs: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Computes reliability table and Expected Calibration Error (ECE)."""
        valid = (~np.isnan(y_true_binary)) & (~np.isnan(pred_probs))
        y = y_true_binary[valid].astype(int)
        p = pred_probs[valid]
        
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(p, bins) - 1
        
        table = []
        ece = 0.0
        n_total = len(y)
        
        for b in range(n_bins):
            mask = (bin_indices == b)
            if np.sum(mask) > 0:
                bin_size = int(np.sum(mask))
                avg_pred = float(np.mean(p[mask]))
                avg_true = float(np.mean(y[mask]))
                table.append({
                    "bin": b + 1,
                    "bin_range": f"[{bins[b]:.2f}, {bins[b+1]:.2f})",
                    "count": bin_size,
                    "mean_predicted_prob": round(avg_pred, 4),
                    "empirical_accuracy": round(avg_true, 4),
                    "abs_calibration_error": round(abs(avg_pred - avg_true), 4)
                })
                ece += (bin_size / n_total) * abs(avg_pred - avg_true)
                
        bs = float(np.mean((p - y) ** 2))
        bs_ref = float(np.mean(y) * (1.0 - np.mean(y)))
        bss = float(1.0 - (bs / (bs_ref + 1e-9)))
        
        return {
            "brier_score": bs,
            "reference_brier_score": bs_ref,
            "brier_skill_score": bss,
            "expected_calibration_error": ece,
            "reliability_table": table,
            "gate_passed": bool(bss > 0.0)
        }
