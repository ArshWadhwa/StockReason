"""Model vs Baseline Comparison & Hurdle Assessment Module."""

from typing import Dict, List, Any, Optional
import pandas as pd


def build_model_comparison_table(
    experiment_results: Dict[str, Any],
    no_change_key: str = "no_change",
    hurdles: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Builds structured comparison DataFrame evaluating each model against No-Change baseline:
    - Relative Error Reduction (vs No-Change): (MAPE_base - MAPE_model) / MAPE_base * 100
    - Predeclared Hurdle Assessment: 2.0% minimum, 5.0% strong, 10.0% stretch.
    """
    if hurdles is None:
        hurdles = {"minimum": 2.0, "strong": 5.0, "stretch": 10.0}

    nc = experiment_results.get(no_change_key, {}).get("diag", {})
    base_mape = nc.get("true_price_mape", 8.7648)
    base_mae = nc.get("inr_price_mae", 103.40)

    rows = []
    for model_id, res in experiment_results.items():
        m_diag = res.get("diag", {})
        if not m_diag:
            continue

        model_mape = m_diag.get("true_price_mape", float("nan"))
        model_mae = m_diag.get("inr_price_mae", float("nan"))

        rel_red = ((base_mape - model_mape) / base_mape * 100.0) if base_mape > 0 else 0.0

        if rel_red >= hurdles["stretch"]:
            hurdle_status = "EXCEEDED_STRETCH (>=10%)"
        elif rel_red >= hurdles["strong"]:
            hurdle_status = "MET_STRONG (>=5%)"
        elif rel_red >= hurdles["minimum"]:
            hurdle_status = "MET_MINIMUM (>=2%)"
        else:
            hurdle_status = "SUB_HURDLE (<2%)"

        rows.append({
            "model_id": model_id,
            "name": res.get("name", model_id),
            "eval_observations": m_diag.get("n_obs", 0),
            "true_price_mape": model_mape,
            "inr_price_mae": model_mae,
            "relative_error_reduction_pct": round(rel_red, 3),
            "directional_accuracy_pct": m_diag.get("directional_accuracy", 0.0),
            "mean_daily_rank_ic": m_diag.get("mean_rank_ic", 0.0),
            "rank_ic_ir": m_diag.get("rank_ic_ir", 0.0),
            "hurdle_assessment": hurdle_status,
        })

    return pd.DataFrame(rows)
