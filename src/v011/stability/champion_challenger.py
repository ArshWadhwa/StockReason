"""
StockReason v0.11 — Champion-Challenger Model Promotion Protocol.
=================================================================
Remediates Phase 6 (Section 41) & Phase 10 (Section 60):
1. Versioned Model Lifecycle States:
   - RESEARCH
   - EXPERIMENTAL
   - HISTORICALLY_VALIDATED
   - SHADOW_TESTING
   - PROSPECTIVELY_VALIDATED
   - PRODUCTION_APPROVED
   - REJECTED
   - RETIRED
2. Non-Negotiable Promotion Gates:
   Gate 1: Critical Data Integrity (0% future leak, no unadjusted gaps).
   Gate 2: Statistical Significance vs Incumbent Champion (paired bootstrap CI < 0).
   Gate 3: Practical Significance (relative loss reduction >= 2.0%).
   Gate 4: Uncertainty Calibration (Quantile error <= 4.0%, Brier Skill > 0).
   Gate 5: Genuine Prospective Track Record.
"""

from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import numpy as np


class ModelLifecycleState(str, Enum):
    RESEARCH = "RESEARCH"
    EXPERIMENTAL = "EXPERIMENTAL"
    HISTORICALLY_VALIDATED = "HISTORICALLY_VALIDATED"
    SHADOW_TESTING = "SHADOW_TESTING"
    PROSPECTIVELY_VALIDATED = "PROSPECTIVELY_VALIDATED"
    PRODUCTION_APPROVED = "PRODUCTION_APPROVED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


def evaluate_champion_challenger_promotion(
    champion_name: str,
    champion_mae: float,
    challenger_name: str,
    challenger_mae: float,
    ci_low: float,
    ci_high: float,
    is_statistically_significant: bool,
    has_data_integrity_pass: bool = True,
    has_calibration_pass: bool = True,
    has_prospective_pass: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates whether a challenger model qualifies to depose the incumbent champion.
    """
    rel_improvement = (champion_mae - challenger_mae) / champion_mae * 100.0

    # Strict promotion checklist
    gate1 = has_data_integrity_pass
    gate2 = challenger_mae < champion_mae and is_statistically_significant and ci_high < 0.0
    gate3 = rel_improvement >= 2.0
    gate4 = has_calibration_pass
    gate5 = has_prospective_pass

    all_passed = gate1 and gate2 and gate3 and gate4 and gate5

    if all_passed:
        verdict = f"PROMOTED: {challenger_name} deposes {champion_name}"
        final_state = ModelLifecycleState.PRODUCTION_APPROVED
    elif gate1 and challenger_mae <= champion_mae and gate4:
        verdict = f"EXPERIMENTAL: {challenger_name} validated historically but pending prospective validation"
        final_state = ModelLifecycleState.HISTORICALLY_VALIDATED
    else:
        verdict = f"REJECTED: {challenger_name} does not meet promotion criteria. Retaining {champion_name}."
        final_state = ModelLifecycleState.REJECTED

    return {
        "champion": champion_name,
        "champion_mae": round(champion_mae, 4),
        "challenger": challenger_name,
        "challenger_mae": round(challenger_mae, 4),
        "relative_improvement_pct": round(rel_improvement, 2),
        "bootstrap_ci_95": f"[{ci_low:.5f}, {ci_high:.5f}]",
        "gate_1_data_integrity": gate1,
        "gate_2_statistical_superiority": gate2,
        "gate_3_practical_magnitude": gate3,
        "gate_4_risk_calibration": gate4,
        "gate_5_prospective_evidence": gate5,
        "final_state": final_state.value,
        "verdict": verdict,
    }
