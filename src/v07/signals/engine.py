"""
StockReason v0.7 — Signal Engine v2
====================================
Evidence-based, calibrated, and cost-aware signal engine for BUY, HOLD, and AVOID decisions.

Key architectural improvements over v0.6 heuristic engine:
1. Calibrated probability floor: Strict enforcement that no BUY signal is emitted
   if calibrated probability < floor (e.g., 0.52).
2. Explicit net-of-cost hurdle: Models transaction friction (brokerage, slippage, STT)
   so expected net return = expected gross return - round-trip friction.
3. Cross-sectional ranking: Tickers compete cross-sectionally within each decision date,
   ensuring BUY signals are allocated to the top relative conviction deciles.
4. Regime-aware hurdle adjustment: Elevated return/probability thresholds during volatile
   or choppy regimes to prevent capital bleeding in hostile market conditions.
5. Genuine model consensus: Down-weights or vetoes setups with model disagreement.
6. Validation-fit thresholds: Calibration and decision parameters can be learned
   from validation folds with zero forward test leakage.
7. Preserves legacy RuleBasedSignalEngine as HeuristicSignalEngine for clean benchmarking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class SignalEngineConfig:
    """Configuration parameters for SignalEngineV2."""
    # Transaction cost model (round-trip basis points, 1 bps = 0.0001)
    # 20 bps = 0.20% (approx 10 bps slippage + 5 bps brokerage/exchange + 5 bps STT/taxes)
    cost_bps: float = 20.0
    
    # Probability thresholds
    # HARD FLOOR: Calibrated probability below this threshold CANNOT generate a BUY signal
    min_calibrated_prob_buy: float = 0.52
    avoid_calibrated_prob: float = 0.46
    
    # Expected net return thresholds (after subtracting costs)
    min_net_return_buy: float = 0.015    # +1.5% net expected return for 1M horizon
    max_net_return_avoid: float = -0.010  # -1.0% net expected return
    
    # Cross-sectional rank percentiles (0.0 to 1.0, where 1.0 is highest ranked in universe)
    min_rank_percentile_buy: float = 0.65  # Top 35% of cross-sectional universe
    max_rank_percentile_avoid: float = 0.35  # Bottom 35% of cross-sectional universe
    
    # Disagreement penalty
    # If models disagree, additional net return hurdle required for BUY
    disagreement_hurdle_boost: float = 0.015  # Requires +1.5% additional return if disagree
    allow_buy_on_disagreement: bool = False   # If False, any disagreement downgrades BUY -> HOLD
    
    # Regime-specific net return hurdles
    regime_hurdle_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low_vol_trending": 1.0,
        "high_vol_trending": 1.15,
        "mean_reverting": 1.10,
        "high_vol_choppy": 1.40,
    })
    
    # Signal score scaling
    base_buy_score: float = 65.0
    base_hold_score: float = 50.0
    base_avoid_score: float = 25.0


@dataclass
class SignalResult:
    """Output container for SignalEngineV2 evaluation."""
    ticker: str
    signal: str                        # "BUY", "HOLD", "AVOID"
    calibrated_probability: float
    expected_gross_return: float
    expected_net_return: float
    cross_sectional_rank: float        # Percentile rank in universe [0, 1]
    regime: str
    model_agreement: bool
    signal_score: float                # Conviction score [0, 100]
    transaction_cost: float
    reasons: List[str]
    is_prob_floor_binding: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "calibrated_probability": round(self.calibrated_probability, 4),
            "expected_gross_return": round(self.expected_gross_return, 4),
            "expected_net_return": round(self.expected_net_return, 4),
            "cross_sectional_rank": round(self.cross_sectional_rank, 4),
            "regime": self.regime,
            "model_agreement": self.model_agreement,
            "signal_score": round(self.signal_score, 1),
            "transaction_cost": round(self.transaction_cost, 4),
            "reasons": self.reasons,
            "is_prob_floor_binding": self.is_prob_floor_binding,
            "metadata": self.metadata,
        }


class SignalEngineV2:
    """
    Evidence-First Signal Engine (Version 2).
    Generates action decisions (BUY/HOLD/AVOID) based on:
      1. Calibrated probability of directional excess return
      2. Net expected return (penalized for realistic trading friction)
      3. Cross-sectional ranking against the broader universe
      4. Market regime volatility & choppiness hurdles
      5. Multi-model consensus / disagreement
    """

    def __init__(self, config: Optional[SignalEngineConfig] = None):
        self.config = config or SignalEngineConfig()

    @property
    def cost_rate(self) -> float:
        """Transaction cost as a decimal fraction (e.g. 20 bps -> 0.0020)."""
        return self.config.cost_bps / 10000.0

    def evaluate_single(
        self,
        ticker: str,
        calibrated_probability: float,
        expected_return: float,
        cross_sectional_rank: float,
        regime: str = "low_vol_trending",
        model_agreement: bool = True,
        is_net_return: bool = False,
        volatility_20d: Optional[float] = None,
        sentiment_score: Optional[float] = None,
        current_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalResult:
        """
        Evaluate a single stock at a single timestamp.

        Parameters:
            ticker: Stock symbol (e.g. "RELIANCE.NS")
            calibrated_probability: Directional probability [0, 1]
            expected_return: Forecasted return (gross by default, or net if is_net_return=True)
            cross_sectional_rank: Relative ranking within universe [0.0 = worst, 1.0 = best]
            regime: Current market regime name
            model_agreement: True if ensemble models agree on direction
            is_net_return: If True, expected_return is already net of costs
            volatility_20d: 20-day realized volatility
            sentiment_score: FinBERT sentiment score [-1, +1]
            current_price: Current market price
            metadata: Optional additional metadata dict
        """
        meta = metadata.copy() if metadata else {}
        reasons: List[str] = []

        # 1. Cost & Net Return Calculation
        if is_net_return:
            expected_net = expected_return
            expected_gross = expected_return + self.cost_rate
        else:
            expected_gross = expected_return
            expected_net = expected_gross - self.cost_rate

        # 2. Regime Hurdle Adjustment
        regime_mult = self.config.regime_hurdle_multipliers.get(regime, 1.0)
        effective_buy_hurdle = self.config.min_net_return_buy * regime_mult
        
        # Disagreement hurdle adjustment
        if not model_agreement and self.config.allow_buy_on_disagreement:
            effective_buy_hurdle += self.config.disagreement_hurdle_boost

        # 3. Decision Logic Evaluation
        prob_floor_binding = False
        
        # Check hard floor: if prob < floor, BUY is strictly disqualified
        if calibrated_probability < self.config.min_calibrated_prob_buy:
            prob_floor_binding = True

        # Preliminary BUY conditions:
        # A) Calibrated prob clears floor
        # B) Net return clears regime-adjusted hurdle
        # C) Cross-sectional rank is in upper tier
        # D) Model agreement holds (or allowed with extra hurdle)
        meets_buy_criteria = (
            not prob_floor_binding and
            expected_net >= effective_buy_hurdle and
            cross_sectional_rank >= self.config.min_rank_percentile_buy and
            (model_agreement or self.config.allow_buy_on_disagreement)
        )

        # Preliminary AVOID conditions:
        meets_avoid_criteria = (
            expected_net <= self.config.max_net_return_avoid or
            calibrated_probability <= self.config.avoid_calibrated_prob or
            (cross_sectional_rank <= self.config.max_rank_percentile_avoid and expected_net < 0.0) or
            (regime == "high_vol_choppy" and expected_net <= 0.002)
        )

        # 4. Final Action Determination & Scoring
        if meets_buy_criteria:
            signal = "BUY"
            # Signal score scales with probability, net return, and rank
            prob_factor = (calibrated_probability - self.config.min_calibrated_prob_buy) * 100.0
            ret_factor = expected_net * 200.0
            rank_factor = (cross_sectional_rank - self.config.min_rank_percentile_buy) * 30.0
            signal_score = min(98.0, max(self.config.base_buy_score,
                                        self.config.base_buy_score + prob_factor + ret_factor + rank_factor))
            
            reasons.append(
                f"Calibrated probability {calibrated_probability:.1%} clears floor "
                f"(>={self.config.min_calibrated_prob_buy:.1%})."
            )
            reasons.append(
                f"Expected net return {expected_net*100:+.2f}% clears regime-adjusted hurdle "
                f"({effective_buy_hurdle*100:.2f}% net after {self.config.cost_bps:.0f} bps cost)."
            )
            reasons.append(
                f"Cross-sectional rank {cross_sectional_rank:.1%} is in top tier "
                f"(>={self.config.min_rank_percentile_buy:.1%})."
            )
            if model_agreement:
                reasons.append("Multi-model consensus confirms directional alignment.")
            else:
                reasons.append("Models exhibit divergence, but return clears elevated hurdle.")
                signal_score *= 0.85

        elif meets_avoid_criteria:
            signal = "AVOID"
            prob_penalty = max(0.0, (self.config.avoid_calibrated_prob - calibrated_probability) * 80.0)
            ret_penalty = max(0.0, -expected_net * 150.0)
            signal_score = max(5.0, min(self.config.base_avoid_score + 10.0,
                                        self.config.base_avoid_score - prob_penalty - ret_penalty))
            
            if expected_net <= self.config.max_net_return_avoid:
                reasons.append(f"Net expected return {expected_net*100:+.2f}% below avoid threshold ({self.config.max_net_return_avoid*100:+.1f}%).")
            if calibrated_probability <= self.config.avoid_calibrated_prob:
                reasons.append(f"Calibrated probability {calibrated_probability:.1%} below avoid floor ({self.config.avoid_calibrated_prob:.1%}).")
            if cross_sectional_rank <= self.config.max_rank_percentile_avoid and expected_net < 0:
                reasons.append(f"Cross-sectional rank {cross_sectional_rank:.1%} in bottom quartile with negative expected return.")
            if regime == "high_vol_choppy":
                reasons.append("Hostile choppy regime renders marginal trade expectancy negative.")
        else:
            signal = "HOLD"
            # Intermediate scoring
            ret_contrib = expected_net * 80.0
            signal_score = max(35.0, min(64.0, self.config.base_hold_score + ret_contrib))
            
            if prob_floor_binding and expected_net >= effective_buy_hurdle:
                reasons.append(
                    f"Expected return {expected_net*100:+.2f}% is attractive, but calibrated probability "
                    f"{calibrated_probability:.1%} is below BUY floor ({self.config.min_calibrated_prob_buy:.1%}). Vetoed to HOLD."
                )
            elif not model_agreement and not self.config.allow_buy_on_disagreement and expected_net >= effective_buy_hurdle:
                reasons.append("Model disagreement vetoes BUY down to HOLD to protect capital.")
            elif cross_sectional_rank < self.config.min_rank_percentile_buy:
                reasons.append(
                    f"Cross-sectional rank {cross_sectional_rank:.1%} fails to qualify for top-tier capital allocation."
                )
            else:
                reasons.append(
                    f"Expectancy ({expected_net*100:+.2f}% net, {calibrated_probability:.1%} prob) favors neutral HOLD."
                )

        if sentiment_score is not None:
            meta["sentiment_score"] = sentiment_score
            if sentiment_score < -0.20 and signal == "BUY":
                # Severe sentiment headwind can also downgrade BUY to HOLD
                signal = "HOLD"
                signal_score = min(signal_score, 55.0)
                reasons.append(f"Adverse sentiment score ({sentiment_score:+.2f}) vetoes aggressive long exposure.")

        if volatility_20d is not None:
            meta["volatility_20d"] = volatility_20d
        if current_price is not None:
            meta["current_price"] = current_price

        return SignalResult(
            ticker=ticker,
            signal=signal,
            calibrated_probability=calibrated_probability,
            expected_gross_return=expected_gross,
            expected_net_return=expected_net,
            cross_sectional_rank=cross_sectional_rank,
            regime=regime,
            model_agreement=model_agreement,
            signal_score=signal_score,
            transaction_cost=self.cost_rate,
            reasons=reasons,
            is_prob_floor_binding=prob_floor_binding,
            metadata=meta,
        )

    def evaluate_universe(
        self,
        df_universe: pd.DataFrame,
        prob_col: str = "calibrated_probability",
        return_col: str = "expected_return",
        regime: str = "low_vol_trending",
        model_agreement_col: Optional[str] = None,
        is_net_return: bool = False,
    ) -> List[SignalResult]:
        """
        Evaluate an entire universe of tickers at a single time step.
        Computes cross-sectional ranks across the universe before generating signals.

        Parameters:
            df_universe: DataFrame with tickers as column or index, plus prob and return cols.
            prob_col: Column name for calibrated probability
            return_col: Column name for expected return
            regime: Current market regime
            model_agreement_col: Column name indicating model agreement (bool)
            is_net_return: Whether return_col is already net of costs
        """
        df = df_universe.copy()
        if "ticker" not in df.columns and "ticker" in df.index.names:
            df = df.reset_index()

        n = len(df)
        if n == 0:
            return []

        # Compute cross-sectional ranks (0.0 to 1.0 percentile rank)
        if n == 1:
            df["cross_sectional_rank"] = 0.5
        else:
            # Rank by composite score of return and probability
            # Standardized ranking based on expected net return
            df["cross_sectional_rank"] = df[return_col].rank(pct=True)

        results = []
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "UNKNOWN"))
            prob = float(row[prob_col])
            ret = float(row[return_col])
            rank = float(row["cross_sectional_rank"])
            agreement = bool(row[model_agreement_col]) if model_agreement_col and model_agreement_col in row else True
            
            vol = float(row["volatility_20d"]) if "volatility_20d" in row and pd.notna(row["volatility_20d"]) else None
            sent = float(row["sentiment_score"]) if "sentiment_score" in row and pd.notna(row["sentiment_score"]) else None
            price = float(row["close"]) if "close" in row and pd.notna(row["close"]) else None

            res = self.evaluate_single(
                ticker=ticker,
                calibrated_probability=prob,
                expected_return=ret,
                cross_sectional_rank=rank,
                regime=regime,
                model_agreement=agreement,
                is_net_return=is_net_return,
                volatility_20d=vol,
                sentiment_score=sent,
                current_price=price,
                metadata={"sector": row.get("sector", "Unknown")} if "sector" in row else None,
            )
            results.append(res)

        return results

    def fit_validation_parameters(
        self,
        val_df: pd.DataFrame,
        prob_col: str = "calibrated_probability",
        return_col: str = "expected_return",
        target_realized_col: str = "realized_return",
    ) -> SignalEngineConfig:
        """
        Fits optimal decision thresholds exclusively on validation folds.
        Ensures ZERO test leakage by grid searching / optimizing Sharpe net of costs on val_df.

        Constraints enforced during tuning:
        1. min_calibrated_prob_buy must be >= 0.50 (no buying sub-50% probability bets).
        2. Minimum net expected return must be > 0.
        """
        best_sharpe = -float("inf")
        best_config = self.config

        # Candidate parameter grid for validation tuning
        candidate_probs = [0.50, 0.52, 0.54, 0.56, 0.58]
        candidate_ranks = [0.60, 0.65, 0.70, 0.75]
        candidate_net_returns = [0.010, 0.015, 0.020, 0.025]

        for p_floor in candidate_probs:
            for r_floor in candidate_ranks:
                for ret_thresh in candidate_net_returns:
                    cfg = SignalEngineConfig(
                        cost_bps=self.config.cost_bps,
                        min_calibrated_prob_buy=p_floor,
                        min_rank_percentile_buy=r_floor,
                        min_net_return_buy=ret_thresh,
                    )
                    engine = SignalEngineV2(cfg)
                    
                    # Evaluate on validation data
                    strat_returns = []
                    for _, row in val_df.iterrows():
                        prob = float(row[prob_col])
                        exp_ret = float(row[return_col])
                        realized = float(row[target_realized_col])
                        rank = float(row.get("cross_sectional_rank", 0.7))
                        
                        res = engine.evaluate_single(
                            ticker=str(row.get("ticker", "T")),
                            calibrated_probability=prob,
                            expected_return=exp_ret,
                            cross_sectional_rank=rank,
                        )
                        # Net realized return for trade
                        if res.signal == "BUY":
                            strat_returns.append(realized - engine.cost_rate)
                        elif res.signal == "HOLD":
                            strat_returns.append(0.5 * realized)
                        else:  # AVOID
                            strat_returns.append(0.0)

                    if len(strat_returns) > 5:
                        mean_r = np.mean(strat_returns)
                        std_r = np.std(strat_returns) + 1e-6
                        sharpe = (mean_r / std_r) * np.sqrt(252)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_config = cfg

        self.config = best_config
        return best_config


# ── Preserved Legacy Engine for Direct Side-by-Side Benchmarking ─────────────
try:
    from src.m4_dashboard.backend.signal_engine import RuleBasedSignalEngine as HeuristicSignalEngine
except ImportError:
    # Standalone fallback if dashboard backend path not resolved
    class HeuristicSignalEngine:
        def __init__(
            self,
            buy_return_threshold: float = 0.025,
            avoid_return_threshold: float = -0.015,
            min_confidence_buy: float = 0.65,
            min_sentiment_buy: float = -0.10,
            disagreement_delta_threshold: float = 0.05,
        ):
            self.buy_return_threshold = buy_return_threshold
            self.avoid_return_threshold = avoid_return_threshold
            self.min_confidence_buy = min_confidence_buy
            self.min_sentiment_buy = min_sentiment_buy
            self.disagreement_delta_threshold = disagreement_delta_threshold

        def evaluate(self, ticker: str, current_price: float, expected_return_1M: float,
                     confidence: float, sentiment_score: float, **kwargs) -> Dict[str, Any]:
            is_buy = (
                expected_return_1M >= self.buy_return_threshold and
                confidence >= self.min_confidence_buy and
                sentiment_score >= self.min_sentiment_buy
            )
            is_avoid = (
                expected_return_1M <= self.avoid_return_threshold or
                sentiment_score <= -0.30
            )
            sig = "BUY" if is_buy else ("AVOID" if is_avoid else "HOLD")
            return {
                "ticker": ticker,
                "signal": sig,
                "current_price": current_price,
                "expected_return_1M": expected_return_1M,
                "confidence": confidence,
                "sentiment_score": sentiment_score,
            }
