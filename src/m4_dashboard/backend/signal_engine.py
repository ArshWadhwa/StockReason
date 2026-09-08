from typing import Dict, Any, List, Optional
import math

class RuleBasedSignalEngine:
    """
    Rule-Based Decision Engine with Real Ensemble Disagreement & Entry/Exit Brackets.
    Combines multi-horizon return, Monte Carlo Dropout confidence, FinBERT sentiment,
    technicals (moving averages, RSI), and genuine two-model ensemble agreement
    into transparent Buy / Hold / Avoid signals with actionable entry/exit timing.
    """

    def __init__(
        self,
        buy_return_threshold: float = 0.025,       # +2.5% 1-Month expected return
        avoid_return_threshold: float = -0.015,     # -1.5% 1-Month expected return
        min_confidence_buy: float = 0.65,          # 65% MC Dropout minimum confidence
        min_sentiment_buy: float = -0.10,          # FinBERT sentiment threshold
        disagreement_delta_threshold: float = 0.05  # 5% LSTM vs secondary model divergence
    ):
        self.buy_return_threshold = buy_return_threshold
        self.avoid_return_threshold = avoid_return_threshold
        self.min_confidence_buy = min_confidence_buy
        self.min_sentiment_buy = min_sentiment_buy
        self.disagreement_delta_threshold = disagreement_delta_threshold

    def evaluate(
        self,
        ticker: str,
        current_price: float,
        expected_return_1M: float,
        confidence: float,
        sentiment_score: float,
        regime: str = "low_vol_trending",
        lstm_return_1M: Optional[float] = None,
        secondary_return_1M: Optional[float] = None,
        sma_50: Optional[float] = None,
        rsi_14: Optional[float] = None,
        atr_14: Optional[float] = None,
        natr_14: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates signal, confidence score, entry/exit brackets,
        and explainable decision trail.
        """
        metadata = metadata or {}
        reasoning: List[str] = []

        # 1. Real Ensemble Disagreement Check
        if lstm_return_1M is not None and secondary_return_1M is not None:
            disagreement_delta = abs(lstm_return_1M - secondary_return_1M)
            disagreement_detected = disagreement_delta > self.disagreement_delta_threshold
        else:
            disagreement_delta = 0.0
            disagreement_detected = False

        if disagreement_detected:
            reasoning.append(
                f"Model Disagreement: Primary LSTM ({lstm_return_1M*100:+.1f}%) diverges from "
                f"Secondary LSTM ({secondary_return_1M*100:+.1f}%) by {disagreement_delta*100:.1f}%. "
                f"Conviction lowered."
            )

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
            # Lower conviction when models disagree
            if disagreement_detected:
                signal_score *= 0.85
            reasoning.append(f"Projected 1-Month return of {expected_return_1M*100:+.2f}% clears buy threshold (>{self.buy_return_threshold*100:.1f}%)")
            reasoning.append(f"Monte Carlo Dropout confidence is robust at {int(confidence*100)}%")
            reasoning.append(f"FinBERT news sentiment confirms positive outlook (Score: {sentiment_score:+.2f})")
            if above_sma50:
                reasoning.append("Price holds above 50-day SMA, confirming medium-term technical uptrend")
            if not disagreement_detected and secondary_return_1M is not None:
                reasoning.append("Primary and Secondary LSTM models are in consensus")

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

        # 4. Entry/Exit Bracket Computation (Issue #12)
        entry_exit = self._compute_entry_exit(
            signal=signal,
            current_price=current_price,
            expected_return_1M=expected_return_1M,
            atr_14=atr_14,
            natr_14=natr_14,
            rsi_14=rsi_14,
            above_sma50=above_sma50,
        )

        if entry_exit and signal == "BUY":
            reasoning.append(
                f"Entry zone: ₹{entry_exit['entry_price']:.2f} | "
                f"Stop-loss: ₹{entry_exit['stop_loss']:.2f} | "
                f"Target: ₹{entry_exit['take_profit']:.2f} "
                f"(Risk:Reward = 1:{entry_exit['risk_reward_ratio']:.1f})"
            )

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
            "reasoning": reasoning,
            "entry_exit": entry_exit,
        }

    def _compute_entry_exit(
        self,
        signal: str,
        current_price: float,
        expected_return_1M: float,
        atr_14: Optional[float],
        natr_14: Optional[float],
        rsi_14: Optional[float],
        above_sma50: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute ATR-based entry/exit brackets for actionable trade execution.
        Returns None if ATR data is unavailable.
        """
        if atr_14 is None or atr_14 <= 0 or current_price <= 0:
            return None

        # ATR ratio (percentage of price)
        atr_ratio = atr_14 / current_price

        if signal == "BUY":
            # Entry: slight pullback from current price (0.5 × ATR)
            entry_price = current_price * (1 - 0.5 * atr_ratio)
            # Stop-loss: 2 × ATR below entry
            stop_loss = entry_price * (1 - 2.0 * atr_ratio)
            # Take-profit: expected return × 1.5 above entry
            take_profit = entry_price * (1 + abs(expected_return_1M) * 1.5)

            # Adjust entry if RSI is oversold (more aggressive entry)
            if rsi_14 and rsi_14 < 35:
                entry_price = current_price * (1 - 0.3 * atr_ratio)

        elif signal == "AVOID":
            # For AVOID signals, provide exit levels for existing positions
            entry_price = current_price  # Exit now
            stop_loss = current_price * (1 - 1.5 * atr_ratio)
            take_profit = current_price * (1 + 0.5 * atr_ratio)  # Tight target for quick exit
        else:
            # HOLD: wider brackets
            entry_price = current_price * (1 - 1.0 * atr_ratio)
            stop_loss = current_price * (1 - 2.5 * atr_ratio)
            take_profit = current_price * (1 + abs(expected_return_1M))

        # Risk/Reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0.0

        return {
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward_ratio": round(rr_ratio, 2),
            "atr_14": round(atr_14, 2),
            "bracket_type": "pullback_entry" if signal == "BUY" else ("exit_now" if signal == "AVOID" else "hold_bracket"),
        }
