from typing import Dict, Any, List, Optional
import math

class RuleBasedSignalEngine:
    """
    Week 1 -> Week 3 Rule-Based Decision Engine.
    Combines multi-horizon return, Monte Carlo Dropout confidence, FinBERT sentiment,
    technicals (moving averages, RSI), and ensemble agreement into transparent Buy / Hold / Avoid signals.
    
    This engine is purely decoupled: in Week 1-3 it runs on mock inputs,
    and in Week 4 it swaps directly to real model predictions with zero logic rework.
    """

    def __init__(
        self,
        buy_return_threshold: float = 0.025,       # +2.5% 1-Month expected return
        avoid_return_threshold: float = -0.015,     # -1.5% 1-Month expected return
        min_confidence_buy: float = 0.65,          # 65% MC Dropout minimum confidence
        min_sentiment_buy: float = -0.10           # FinBERT sentiment threshold
    ):
        self.buy_return_threshold = buy_return_threshold
        self.avoid_return_threshold = avoid_return_threshold
        self.min_confidence_buy = min_confidence_buy
        self.min_sentiment_buy = min_sentiment_buy

    def evaluate(
        self,
        ticker: str,
        current_price: float,
        expected_return_1M: float,
        confidence: float,
        sentiment_score: float,
        regime: str = "low_vol_trending",
        lstm_return_1M: Optional[float] = None,
        sma_50: Optional[float] = None,
        rsi_14: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates signal, confidence score, and explainable decision trail.
        """
        metadata = metadata or {}
        reasoning: List[str] = []

        disagreement_detected = False

        # 2. Technical Trend Modifier
        above_sma50 = True if (sma_50 and current_price >= sma_50) else False
        rsi_overbought = True if (rsi_14 and rsi_14 > 75) else False
        rsi_oversold = True if (rsi_14 and rsi_14 < 30) else False

        # 3. Decision Tree Logic
        is_buy = (
            expected_return_1M >= self.buy_return_threshold and
            confidence >= self.min_confidence_buy and
            sentiment_score >= self.min_sentiment_buy and
            not (disagreement_detected and expected_return_1M < 0.04) and
            not rsi_overbought
        )

        is_avoid = (
            expected_return_1M <= self.avoid_return_threshold or
            sentiment_score <= -0.30 or
            (disagreement_detected and confidence < 0.60) or
            (regime == "high_vol_choppy" and expected_return_1M < 0.01)
        )

        if is_buy:
            signal = "BUY"
            signal_score = min(98.0, 60.0 + (expected_return_1M * 250.0) + (confidence * 20.0))
            reasoning.append(f"Projected 1-Month return of {expected_return_1M*100:+.2f}% clears buy threshold (>{self.buy_return_threshold*100:.1f}%)")
            reasoning.append(f"Monte Carlo Dropout confidence is robust at {int(confidence*100)}%")
            reasoning.append(f"FinBERT news sentiment confirms positive outlook (Score: {sentiment_score:+.2f})")
            if above_sma50:
                reasoning.append("Price holds above 50-day SMA, confirming medium-term technical uptrend")

        elif is_avoid:
            signal = "AVOID"
            signal_score = max(5.0, 20.0 + (expected_return_1M * 100.0))
            if expected_return_1M <= self.avoid_return_threshold:
                reasoning.append(f"Negative return forecast: {expected_return_1M*100:+.2f}% below threshold")
            if sentiment_score <= -0.25:
                reasoning.append(f"Adverse sentiment detected: News tone score {sentiment_score:+.2f}")
            if disagreement_detected:
                reasoning.append("Model disagreement creates high structural uncertainty")
            if regime == "high_vol_choppy":
                reasoning.append("High volatility choppy market regime discourages capital commitment")
        else:
            signal = "HOLD"
            signal_score = max(30.0, min(65.0, 48.0 + (expected_return_1M * 120.0)))
            reasoning.append(f"Moderate 1-Month expected return of {expected_return_1M*100:+.2f}%")
            reasoning.append("Confidence / sentiment profile favors maintaining holding position over fresh capital deployment")

        return {
            "ticker": ticker,
            "name": metadata.get("name", ticker),
            "sector": metadata.get("sector", "Unknown"),
            "universe": metadata.get("universe", "NIFTY 50"),
            "current_price": round(current_price, 2),
            "signal": signal,
            "signal_score": round(signal_score, 1),
            "confidence": round(confidence, 2),
            "expected_return_1M": round(expected_return_1M, 4),
            "sentiment_score": round(sentiment_score, 3),
            "regime": regime,
            "ensemble_agreement": not disagreement_detected,
            "reasoning": reasoning
        }
