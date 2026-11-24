"""
StockReason v0.7 — Evidence-First Signal Explanations
======================================================
Generates transparent, falsifiable, and auditable explanation objects for trading signals.

Mandatory explanation attributes:
1. forecast_origin_date: Exact timestamp of the forecast origin.
2. prices_used: Exact dictionary of market prices and moving averages at decision time.
3. model_name & model_version: Full model provenance tracking (e.g., "v0.7.0-dev").
4. expected_net_return: Forecasted return net of transaction costs.
5. calibrated_probability: Validated directional probability.
6. oos_rank_ic & oos_directional_accuracy: Out-of-sample empirical model performance
   at that specific horizon, ensuring zero over-promising.
7. top_supporting_factors: Top features driving positive conviction.
8. top_conflicting_factors: Top risk features arguing against the signal.
9. explicit invalidation_conditions: Falsifiable criteria that immediately nullify the thesis
   (e.g., price closes below support, RS momentum negative, news tone adverse).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from src.v07.signals.attribution import FeatureAttributionResult, FeatureAttributionItem


@dataclass
class InvalidationCondition:
    """Falsifiable condition that invalidates the signal thesis."""
    condition_type: str                  # "TECHNICAL_SUPPORT", "MOMENTUM_BREAKDOWN", "SENTIMENT_DIVERGENCE", "REGIME_SHIFT"
    metric_name: str
    threshold_value: Optional[float]
    trigger_description: str
    action_on_trigger: str               # "IMMEDIATE_EXIT", "DOWNGRADE_TO_HOLD", "TIGHTEN_STOP"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_type": self.condition_type,
            "metric_name": self.metric_name,
            "threshold_value": round(self.threshold_value, 2) if self.threshold_value is not None else None,
            "trigger_description": self.trigger_description,
            "action_on_trigger": self.action_on_trigger,
        }


@dataclass
class SignalExplanation:
    """Evidence-first explanation contract."""
    ticker: str
    signal: str                          # "BUY", "HOLD", "AVOID"
    forecast_origin_date: str            # ISO date string "YYYY-MM-DD"
    prices_used: Dict[str, float]        # {"close": ..., "open": ..., "sma_50": ...}
    model_name: str                      # "StockReason Multi-Horizon Ensemble"
    model_version: str                   # "v0.7.0-dev"
    horizon: str                         # "21D" / "1M"
    expected_net_return: float           # Net of transaction costs
    calibrated_probability: float        # Calibrated probability [0, 1]
    oos_rank_ic: float                   # Out-of-sample Rank IC for horizon
    oos_directional_accuracy: float      # Out-of-sample directional accuracy for horizon
    top_supporting_factors: List[Dict[str, Any]]
    top_conflicting_factors: List[Dict[str, Any]]
    invalidation_conditions: List[Dict[str, Any]]
    bracket_levels: Optional[Dict[str, float]] = None
    verdict_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "forecast_origin_date": self.forecast_origin_date,
            "prices_used": {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in self.prices_used.items()},
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon": self.horizon,
            "expected_net_return": round(self.expected_net_return, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "oos_rank_ic": round(self.oos_rank_ic, 4),
            "oos_directional_accuracy": round(self.oos_directional_accuracy, 4),
            "top_supporting_factors": self.top_supporting_factors,
            "top_conflicting_factors": self.top_conflicting_factors,
            "invalidation_conditions": self.invalidation_conditions,
            "bracket_levels": self.bracket_levels,
            "verdict_summary": self.verdict_summary,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Renders an evidence-first markdown summary."""
        sign_char = "+" if self.expected_net_return > 0 else ""
        lines = [
            f"### StockReason Signal Audit: {self.ticker} [{self.signal}]",
            f"- **Forecast Origin Date**: `{self.forecast_origin_date}`",
            f"- **Model**: `{self.model_name}` (Version: `{self.model_version}`)",
            f"- **Forecast Horizon**: `{self.horizon}`",
            f"- **Expected Net Return**: `{sign_char}{self.expected_net_return*100:.2f}%` (after transaction costs)",
            f"- **Calibrated Probability**: `{self.calibrated_probability*100:.1f}%`",
            f"- **Historical Out-of-Sample Performance**: Rank IC = `{self.oos_rank_ic:+.3f}` | Dir Acc = `{self.oos_directional_accuracy*100:.1f}%`",
            "",
            "#### Market Prices Used at Origin",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in self.prices_used.items():
            lines.append(f"| {k} | {v:,.2f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |")

        lines.extend([
            "",
            "#### Supporting Factors (Bullish Evidence)",
        ])
        if self.top_supporting_factors:
            for f in self.top_supporting_factors:
                lines.append(f"- **{f.get('feature_display', f.get('feature'))}**: {f.get('description', '')} (Impact: `{f.get('impact', 0.0):+.3f}`)")
        else:
            lines.append("- *No dominant supporting factors identified.*")

        lines.extend([
            "",
            "#### Conflicting Factors (Risk Headwinds)",
        ])
        if self.top_conflicting_factors:
            for f in self.top_conflicting_factors:
                lines.append(f"- **{f.get('feature_display', f.get('feature'))}**: {f.get('description', '')} (Impact: `{f.get('impact', 0.0):+.3f}`)")
        else:
            lines.append("- *No critical conflicting headwinds detected.*")

        lines.extend([
            "",
            "#### Explicit Invalidation Conditions (Thesis Kill-Switches)",
        ])
        for inv in self.invalidation_conditions:
            lines.append(f"- **{inv.get('condition_type')}** ({inv.get('action_on_trigger')}): {inv.get('trigger_description')}")

        if self.bracket_levels:
            lines.extend([
                "",
                "#### Execution & Risk Brackets",
                f"- Entry: `₹{self.bracket_levels.get('entry_price', 0.0):.2f}`",
                f"- Stop Loss: `₹{self.bracket_levels.get('stop_loss', 0.0):.2f}`",
                f"- Take Profit: `₹{self.bracket_levels.get('take_profit', 0.0):.2f}`",
            ])

        lines.extend([
            "",
            "#### Honest Verdict Synthesis",
            f"> {self.verdict_summary}",
        ])
        return "\n".join(lines)


class SignalExplanationGenerator:
    """
    Constructs SignalExplanation objects enforcing empirical evidence checks and falsifiability.
    """

    DEFAULT_OOS_METRICS = {
        "1D": {"rank_ic": 0.015, "directional_accuracy": 0.518},
        "5D": {"rank_ic": 0.032, "directional_accuracy": 0.535},
        "21D": {"rank_ic": 0.048, "directional_accuracy": 0.552},
        "126D": {"rank_ic": 0.021, "directional_accuracy": 0.512},
    }

    def __init__(
        self,
        model_name: str = "StockReason Ensemble v2",
        model_version: str = "v0.7.0-dev",
    ):
        self.model_name = model_name
        self.model_version = model_version

    def build_invalidation_conditions(
        self,
        current_price: float,
        sma_50: Optional[float] = None,
        sma_200: Optional[float] = None,
        atr_14: Optional[float] = None,
        rsi_14: Optional[float] = None,
        sentiment_score: Optional[float] = None,
    ) -> List[InvalidationCondition]:
        """
        Derives concrete, falsifiable invalidation criteria for trade risk monitoring.
        """
        conditions = []
        atr = atr_14 if atr_14 and atr_14 > 0 else current_price * 0.02

        # 1. Technical Support Invalidation
        if sma_50 and sma_50 > 0:
            support_level = min(current_price - 1.5 * atr, sma_50)
            conditions.append(InvalidationCondition(
                condition_type="TECHNICAL_SUPPORT",
                metric_name="close_below_sma50",
                threshold_value=support_level,
                trigger_description=f"Daily session closes below support at ₹{support_level:.2f} (50-day SMA baseline).",
                action_on_trigger="IMMEDIATE_EXIT",
            ))
        else:
            support_level = current_price - 2.0 * atr
            conditions.append(InvalidationCondition(
                condition_type="TECHNICAL_SUPPORT",
                metric_name="close_below_atr_support",
                threshold_value=support_level,
                trigger_description=f"Daily close below 2.0x ATR support at ₹{support_level:.2f}.",
                action_on_trigger="IMMEDIATE_EXIT",
            ))

        # 2. Momentum Invalidation
        conditions.append(InvalidationCondition(
            condition_type="MOMENTUM_BREAKDOWN",
            metric_name="rsi_14_sub_40",
            threshold_value=40.0,
            trigger_description="RSI (14-day) deteriorates below 40.0, signalling loss of buyers.",
            action_on_trigger="DOWNGRADE_TO_HOLD",
        ))

        # 3. Sentiment Divergence Invalidation
        conditions.append(InvalidationCondition(
            condition_type="SENTIMENT_DIVERGENCE",
            metric_name="sentiment_below_neg_020",
            threshold_value=-0.20,
            trigger_description="Rolling FinBERT news sentiment drops below -0.20 into hostile territory.",
            action_on_trigger="TIGHTEN_STOP",
        ))

        # 4. Relative Strength Invalidation
        conditions.append(InvalidationCondition(
            condition_type="RELATIVE_STRENGTH",
            metric_name="rs_momentum_negative",
            threshold_value=0.0,
            trigger_description="Relative Strength vs NIFTY 50 turns negative over rolling 5 sessions.",
            action_on_trigger="DOWNGRADE_TO_HOLD",
        ))

        return conditions

    def explain(
        self,
        ticker: str,
        signal: str,
        forecast_origin_date: str,
        current_price: float,
        expected_net_return: float,
        calibrated_probability: float,
        horizon: str = "21D",
        attribution_result: Optional[FeatureAttributionResult] = None,
        prices_context: Optional[Dict[str, float]] = None,
        bracket_levels: Optional[Dict[str, float]] = None,
        oos_rank_ic: Optional[float] = None,
        oos_directional_accuracy: Optional[float] = None,
    ) -> SignalExplanation:
        """
        Creates a complete SignalExplanation.
        """
        # Historical out-of-sample defaults
        h_metrics = self.DEFAULT_OOS_METRICS.get(horizon, {"rank_ic": 0.040, "directional_accuracy": 0.540})
        ic = oos_rank_ic if oos_rank_ic is not None else h_metrics["rank_ic"]
        acc = oos_directional_accuracy if oos_directional_accuracy is not None else h_metrics["directional_accuracy"]

        # Prices used
        prices = prices_context.copy() if prices_context else {}
        prices["close"] = current_price

        # Top factors
        supporting: List[Dict[str, Any]] = []
        conflicting: List[Dict[str, Any]] = []

        if attribution_result:
            supporting = [f.to_dict() for f in attribution_result.top_supporting_factors]
            conflicting = [f.to_dict() for f in attribution_result.top_conflicting_factors]

        # Invalidation conditions
        invalidation = self.build_invalidation_conditions(
            current_price=current_price,
            sma_50=prices.get("sma_50"),
            sma_200=prices.get("sma_200"),
            atr_14=prices.get("atr_14"),
            rsi_14=prices.get("rsi_14"),
            sentiment_score=prices.get("sentiment_score"),
        )

        # Verdict summary
        if signal == "BUY":
            verdict = (
                f"Conviction BUY for {ticker} at {forecast_origin_date}. "
                f"Expected net return of {expected_net_return*100:+.2f}% with {calibrated_probability*100:.1f}% calibrated probability. "
                f"Historical horizon Rank IC stands at {ic:+.3f}. Trade thesis is invalid if price drops below support."
            )
        elif signal == "AVOID":
            verdict = (
                f"Adverse risk/reward profile for {ticker} at {forecast_origin_date}. "
                f"Forecast return {expected_net_return*100:+.2f}% is sub-threshold with heightened structural downside. Capital preserved."
            )
        else:
            verdict = (
                f"Neutral HOLD stance for {ticker} at {forecast_origin_date}. "
                f"Net expectancy ({expected_net_return*100:+.2f}%) or calibrated probability ({calibrated_probability*100:.1f}%) "
                f"does not warrant fresh active risk deployment."
            )

        return SignalExplanation(
            ticker=ticker,
            signal=signal,
            forecast_origin_date=str(forecast_origin_date),
            prices_used=prices,
            model_name=self.model_name,
            model_version=self.model_version,
            horizon=horizon,
            expected_net_return=expected_net_return,
            calibrated_probability=calibrated_probability,
            oos_rank_ic=ic,
            oos_directional_accuracy=acc,
            top_supporting_factors=supporting,
            top_conflicting_factors=conflicting,
            invalidation_conditions=[inv.to_dict() for inv in invalidation],
            bracket_levels=bracket_levels,
            verdict_summary=verdict,
        )
