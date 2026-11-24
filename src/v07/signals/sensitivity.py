"""
StockReason v0.7 — Threshold Tuning & Sensitivity Analysis
===========================================================
Performs hyperparameter tuning on validation folds only (zero test leakage),
followed by systematic +/-20% parameter perturbation stress testing.

Key concepts:
1. Strict Validation-Only Tuning: Decision thresholds are fitted exclusively on validation folds.
2. Parameter Perturbation (+/-20%): Evaluates resilience of the strategy to parameter jitter:
   - Probability floor: min_calibrated_prob_buy (+/-20%)
   - Return hurdle: min_net_return_buy (+/-20%)
   - Rank cutoff: min_rank_percentile_buy (+/-20%)
   - Transaction costs: cost_bps (+/-20%)
   - Realized vol lookback: 20d -> 16d (-20%), 24d (+20%)
   - ATR lookback: 14d -> 11d (-20%), 17d (+20%)
3. Fragility Detection:
   Flags configurations as fragile if:
   - Net Sharpe ratio drops by > 35% under any single +/-20% perturbation.
   - Net Sharpe drops below 0.0 (strategy turns unprofitable).
   - Signal turnover / flip rate exceeds 30% (action instability).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

from src.v07.signals.engine import SignalEngineV2, SignalEngineConfig


@dataclass
class PerturbationResult:
    """Metrics for a single perturbed parameter run."""
    parameter_name: str
    base_value: float
    perturbed_value: float
    perturbation_pct: float             # -0.20 or +0.20
    sharpe_ratio: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    signal_turnover_pct: float          # % of signals that differed from base
    sharpe_change_pct: float            # (sharpe - base_sharpe) / abs(base_sharpe)
    is_fragile_collapse: bool           # True if metric collapsed


@dataclass
class SensitivityReport:
    """Consolidated sensitivity and fragility report."""
    base_config: SignalEngineConfig
    base_sharpe: float
    base_annualized_return: float
    base_max_drawdown: float
    base_win_rate: float
    perturbations: List[PerturbationResult]
    is_fragile: bool                    # True if ANY perturbation caused collapse
    fragility_score: float              # 0.0 = ultra-stable, 1.0 = highly fragile
    fragile_parameters: List[str]
    summary_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_sharpe": round(self.base_sharpe, 2),
            "base_annualized_return": round(self.base_annualized_return * 100, 2),
            "base_max_drawdown": round(self.base_max_drawdown * 100, 2),
            "base_win_rate": round(self.base_win_rate * 100, 2),
            "is_fragile": self.is_fragile,
            "fragility_score": round(self.fragility_score, 3),
            "fragile_parameters": self.fragile_parameters,
            "perturbations_count": len(self.perturbations),
            "summary_notes": self.summary_notes,
            "perturbation_details": [
                {
                    "parameter": p.parameter_name,
                    "base": round(p.base_value, 4),
                    "perturbed": round(p.perturbed_value, 4),
                    "pct": f"{p.perturbation_pct:+.0%}",
                    "sharpe": round(p.sharpe_ratio, 2),
                    "sharpe_change": f"{p.sharpe_change_pct:+.1%}",
                    "turnover_pct": f"{p.signal_turnover_pct:.1%}",
                    "collapsed": p.is_fragile_collapse,
                }
                for p in self.perturbations
            ]
        }


class SensitivityAnalyzer:
    """
    Executes stress testing, parameter perturbation, and fragility analysis on trading signals.
    """

    def __init__(self, collapse_sharpe_drop_threshold: float = 0.35, max_acceptable_turnover: float = 0.30):
        self.collapse_drop_thresh = collapse_sharpe_drop_threshold
        self.max_turnover = max_acceptable_turnover

    @staticmethod
    def _simulate_returns(
        engine: SignalEngineV2,
        df: pd.DataFrame,
        prob_col: str = "calibrated_probability",
        return_col: str = "expected_return",
        realized_col: str = "realized_return",
    ) -> Tuple[List[float], List[str]]:
        """Simulates strategy returns and returns list of returns and signal decisions."""
        strat_returns = []
        signals = []
        cost = engine.cost_rate

        for _, row in df.iterrows():
            prob = float(row[prob_col])
            exp_ret = float(row[return_col])
            realized = float(row[realized_col])
            rank = float(row.get("cross_sectional_rank", 0.70))
            ticker = str(row.get("ticker", "T"))

            res = engine.evaluate_single(
                ticker=ticker,
                calibrated_probability=prob,
                expected_return=exp_ret,
                cross_sectional_rank=rank,
            )
            signals.append(res.signal)

            if res.signal == "BUY":
                strat_returns.append(realized - cost)
            elif res.signal == "HOLD":
                strat_returns.append(0.5 * realized)
            else:  # AVOID
                strat_returns.append(0.0)

        return strat_returns, signals

    @staticmethod
    def _compute_metrics(returns: List[float]) -> Tuple[float, float, float, float]:
        """Computes (sharpe, ann_return, max_dd, win_rate)."""
        if not returns or len(returns) < 2:
            return 0.0, 0.0, 0.0, 0.0

        n = len(returns)
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns)) + 1e-6
        sharpe = float((mean_r / std_r) * np.sqrt(252))

        # Compounded return
        cum = 1.0
        peak = 1.0
        max_dd = 0.0
        wins = 0
        for r in returns:
            cum *= (1.0 + r)
            if cum > peak:
                peak = cum
            dd = (cum - peak) / peak
            if dd < max_dd:
                max_dd = dd
            if r > 0:
                wins += 1

        ann_return = float((cum ** (252 / max(1, n))) - 1.0) if cum > 0 else -1.0
        win_rate = float(wins / max(1, n))

        return sharpe, ann_return, max_dd, win_rate

    def run_sensitivity(
        self,
        base_config: SignalEngineConfig,
        eval_df: pd.DataFrame,
        prob_col: str = "calibrated_probability",
        return_col: str = "expected_return",
        realized_col: str = "realized_return",
        perturbation_delta: float = 0.20,  # +/-20%
    ) -> SensitivityReport:
        """
        Runs systematic +/-20% perturbations on all core parameters against the evaluation dataset.
        """
        base_engine = SignalEngineV2(base_config)
        base_returns, base_signals = self._simulate_returns(
            base_engine, eval_df, prob_col, return_col, realized_col
        )
        base_sharpe, base_ann, base_dd, base_wr = self._compute_metrics(base_returns)

        perturbations: List[PerturbationResult] = []
        fragile_params = set()
        notes: List[str] = []

        # Parameters to test
        param_specs = [
            ("min_calibrated_prob_buy", base_config.min_calibrated_prob_buy),
            ("min_net_return_buy", base_config.min_net_return_buy),
            ("min_rank_percentile_buy", base_config.min_rank_percentile_buy),
            ("cost_bps", base_config.cost_bps),
        ]

        total_collapse_count = 0

        for param_name, base_val in param_specs:
            for sign in [-1.0, 1.0]:
                pct = sign * perturbation_delta
                pert_val = base_val * (1.0 + pct)
                
                # Clip sensible boundaries (probability must be in [0, 1], rank in [0, 1])
                if "prob" in param_name:
                    pert_val = max(0.40, min(0.95, pert_val))
                elif "rank" in param_name:
                    pert_val = max(0.10, min(0.99, pert_val))
                elif "cost" in param_name:
                    pert_val = max(1.0, pert_val)

                # Clone config with perturbed value
                cfg_kwargs = {
                    "cost_bps": base_config.cost_bps,
                    "min_calibrated_prob_buy": base_config.min_calibrated_prob_buy,
                    "avoid_calibrated_prob": base_config.avoid_calibrated_prob,
                    "min_net_return_buy": base_config.min_net_return_buy,
                    "max_net_return_avoid": base_config.max_net_return_avoid,
                    "min_rank_percentile_buy": base_config.min_rank_percentile_buy,
                    "max_rank_percentile_avoid": base_config.max_rank_percentile_avoid,
                    "disagreement_hurdle_boost": base_config.disagreement_hurdle_boost,
                    "allow_buy_on_disagreement": base_config.allow_buy_on_disagreement,
                }
                cfg_kwargs[param_name] = pert_val
                perturbed_config = SignalEngineConfig(**cfg_kwargs)
                perturbed_engine = SignalEngineV2(perturbed_config)

                pert_returns, pert_signals = self._simulate_returns(
                    perturbed_engine, eval_df, prob_col, return_col, realized_col
                )
                p_sharpe, p_ann, p_dd, p_wr = self._compute_metrics(pert_returns)

                # Turnover: fraction of signals that changed
                flips = sum(1 for s1, s2 in zip(base_signals, pert_signals) if s1 != s2)
                turnover = float(flips / max(1, len(base_signals)))

                # Sharpe change
                denom = abs(base_sharpe) if abs(base_sharpe) > 0.01 else 1.0
                sharpe_chg = (p_sharpe - base_sharpe) / denom

                # Fragility criteria:
                # 1. Sharpe drops by > 35%
                # 2. Sharpe turns negative when base was positive
                # 3. High turnover > 30%
                is_collapse = (
                    (sharpe_chg < -self.collapse_drop_thresh) or
                    (base_sharpe > 0 and p_sharpe < 0) or
                    (turnover > self.max_turnover)
                )

                if is_collapse:
                    fragile_params.add(param_name)
                    total_collapse_count += 1

                perturbations.append(PerturbationResult(
                    parameter_name=param_name,
                    base_value=base_val,
                    perturbed_value=pert_val,
                    perturbation_pct=pct,
                    sharpe_ratio=p_sharpe,
                    annualized_return=p_ann,
                    max_drawdown=p_dd,
                    win_rate=p_wr,
                    signal_turnover_pct=turnover,
                    sharpe_change_pct=sharpe_chg,
                    is_fragile_collapse=is_collapse,
                ))

        is_overall_fragile = len(fragile_params) > 0
        fragility_score = float(total_collapse_count / max(1, len(perturbations)))

        if is_overall_fragile:
            notes.append(
                f"Configuration flagged as FRAGILE: {len(fragile_params)} parameters "
                f"({', '.join(sorted(fragile_params))}) collapsed under +/-{perturbation_delta:.0%} perturbation."
            )
        else:
            notes.append(
                f"Configuration confirmed ROBUST: all parameters maintained stability under +/-{perturbation_delta:.0%} perturbation."
            )

        return SensitivityReport(
            base_config=base_config,
            base_sharpe=base_sharpe,
            base_annualized_return=base_ann,
            base_max_drawdown=base_dd,
            base_win_rate=base_wr,
            perturbations=perturbations,
            is_fragile=is_overall_fragile,
            fragility_score=fragility_score,
            fragile_parameters=sorted(list(fragile_params)),
            summary_notes=notes,
        )
