"""Downside Risk Probability & Brier Skill Score Calibration Gate."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import brier_score_loss


def evaluate_probability_calibration(
    y_true_binary: np.ndarray,
    predicted_probs: np.ndarray,
    base_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes Brier Score and Brier Skill Score (BSS):
    BSS = 1 - (Brier_model / Brier_ref)
    where Brier_ref is the uninformative historical base rate.
    Gate: If BSS <= 0, probability forecast is uncalibrated and MUST BE SUPPRESSED.
    """
    if len(y_true_binary) == 0:
        return {"status": "NO_DATA", "brier_skill_score": -1.0, "is_calibrated": False}

    bs_model = float(brier_score_loss(y_true_binary, predicted_probs))
    if base_rate is None:
        base_rate = float(np.mean(y_true_binary))

    ref_probs = np.full_like(predicted_probs, base_rate)
    bs_ref = float(brier_score_loss(y_true_binary, ref_probs))

    bss = 1.0 - (bs_model / bs_ref) if bs_ref > 0 else -1.0
    is_calibrated = (bss > 0.0)

    return {
        "brier_score_model": round(bs_model, 6),
        "brier_score_reference": round(bs_ref, 6),
        "brier_skill_score": round(bss, 4),
        "base_rate": round(base_rate, 4),
        "is_calibrated": is_calibrated,
        "action": "PROCEED" if is_calibrated else "SUPPRESS_PROBABILITY",
    }
