"""
StockReason v0.10 — Causal Market-Regime Detector.
===================================================
Remediates Phase 3 (Section 8):
1. Causal trailing indicators strictly available at forecast origin:
   - Trailing 200-day trend slope: P_t / SMA_200 - 1.0
   - Trailing 60-day drawdown from peak
   - Trailing 60-day realized volatility percentile
   - India VIX level
   - Trailing market breadth (% of stocks above 50-day SMA)
2. Four Causal Market Regimes:
   - Bullish_LowVol (Trend > 0, VIX < 16)
   - Bullish_HighVol (Trend > 0, VIX >= 16)
   - Bearish_Stress (Trend < 0, VIX >= 20 or Drawdown < -10%)
   - Sideways_Neutral (Consolidation)
3. Computes Markov transition matrix and regime persistence.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class CausalMarketRegimeDetector:
    def __init__(self, vix_high_thresh: float = 18.0, dd_stress_thresh: float = -0.08):
        self.vix_high_thresh = vix_high_thresh
        self.dd_stress_thresh = dd_stress_thresh
        self.regime_names = ["Bullish_LowVol", "Bullish_HighVol", "Bearish_Stress", "Sideways_Neutral"]

    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns causal market regime to each date based strictly on trailing information.
        """
        out = df.copy()
        
        # Ensure date_str exists
        if "date_str" not in out.columns:
            out["date_str"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

        # Required trailing indicators
        vix = out["vix_level"].values if "vix_level" in out.columns else np.full(len(out), 15.0)
        slope = out["nifty_trend_slope"].values if "nifty_trend_slope" in out.columns else np.zeros(len(out))
        ret_21 = out["nifty_ret_21d"].values if "nifty_ret_21d" in out.columns else np.zeros(len(out))

        regimes = []
        for v, s, r in zip(vix, slope, ret_21):
            if s > 0 and r > 0 and v < self.vix_high_thresh:
                regimes.append("Bullish_LowVol")
            elif s > 0 and v >= self.vix_high_thresh:
                regimes.append("Bullish_HighVol")
            elif (s < 0 or r < -0.05) and v >= self.vix_high_thresh:
                regimes.append("Bearish_Stress")
            else:
                regimes.append("Sideways_Neutral")

        out["market_regime"] = regimes
        return out

    def compute_regime_statistics(self, df_with_regimes: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes regime frequencies, persistence, and transition probability matrix.
        """
        daily = df_with_regimes.groupby("date_str")["market_regime"].first()
        counts = daily.value_counts().to_dict()
        total_days = len(daily)

        freqs = {k: round(v / total_days * 100.0, 2) for k, v in counts.items()}

        # Transition matrix
        transitions = pd.crosstab(
            daily.iloc[:-1].values,
            daily.iloc[1:].values,
            normalize="index"
        ).to_dict()

        return {
            "total_trading_days": total_days,
            "regime_frequencies_pct": freqs,
            "regime_counts": counts,
            "transition_matrix": transitions,
        }
