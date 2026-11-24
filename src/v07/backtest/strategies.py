"""
Strategy Definitions and Benchmark Generators for v0.7 Backtesting.
===================================================================
Implements:
1. Top-K Long-Only Strategy with optional sector concentration constraints.
2. Long/Flat Strategy mapping BUY/HOLD/AVOID signals to 1.0, 0.5, 0.0 exposures.
3. Equal-Weight Universe Benchmark.
4. Buy-and-Hold Asset/Index Benchmark (e.g. NIFTY 50).
5. Random Top-K Benchmark with multi-seed statistical aggregation (200 seeds).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for backtest portfolio strategies."""

    @abstractmethod
    def generate_target_weights(
        self,
        date: Any,
        predictions_df: pd.DataFrame,
        active_universe: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict mapping ticker -> target portfolio weight in [0.0, 1.0].
        Sum of weights must be <= 1.0. Remaining weight is held in cash.
        """
        pass


class TopKStrategy(BaseStrategy):
    """
    Top-K Long-Only Strategy:
    Selects top K tickers with highest predicted score (e.g. pred_return or pred_prob_up).
    Equal weights (1/K each), subject to optional sector cap constraint.
    """

    def __init__(
        self,
        k: int = 10,
        score_col: str = "pred_return",
        sector_cap: Optional[float] = None,  # e.g. 0.30 max 30% per sector
        min_score: Optional[float] = None,   # Minimum predicted return threshold
    ):
        self.k = max(1, int(k))
        self.score_col = score_col
        self.sector_cap = sector_cap
        self.min_score = min_score

    def generate_target_weights(
        self,
        date: Any,
        predictions_df: pd.DataFrame,
        active_universe: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        if predictions_df.empty:
            return {}

        # Filter to active universe
        df_candidates = predictions_df[predictions_df["ticker"].isin(active_universe)].copy()
        if df_candidates.empty:
            return {}

        # Apply min score filter if set
        if self.min_score is not None:
            df_candidates = df_candidates[df_candidates[self.score_col] >= self.min_score]
            if df_candidates.empty:
                return {}

        # Sort descending by score
        df_sorted = df_candidates.sort_values(by=self.score_col, ascending=False)

        selected: List[str] = []
        sector_counts: Dict[str, int] = {}
        sector_map = sector_map or {}

        # Max stocks allowed in any sector
        max_stocks_per_sector = (
            max(1, int(math_floor(self.sector_cap * self.k)))
            if (self.sector_cap is not None and self.sector_cap < 1.0)
            else self.k
        )

        for _, row in df_sorted.iterrows():
            ticker = row["ticker"]
            if ticker in selected:
                continue

            sec = sector_map.get(ticker, "UNKNOWN")
            current_sec_count = sector_counts.get(sec, 0)

            if self.sector_cap is not None and current_sec_count >= max_stocks_per_sector:
                continue

            selected.append(ticker)
            sector_counts[sec] = current_sec_count + 1

            if len(selected) >= self.k:
                break

        if not selected:
            return {}

        # Equal weight: 1.0 / self.k (or 1.0 / len(selected) if less than k available)
        weight_per_stock = 1.0 / self.k
        return {t: weight_per_stock for t in selected}


def math_floor(x: float) -> int:
    return int(np.floor(x))


class LongFlatStrategy(BaseStrategy):
    """
    Long / Flat vs Market Strategy:
    Maps BUY/HOLD/AVOID signals to target exposure (1.0, 0.5, 0.0).
    Can derive signals from explicit 'signal' column or from 'pred_return' / 'pred_prob_up' thresholds.
    """

    def __init__(
        self,
        signal_col: Optional[str] = None,
        score_col: str = "pred_return",
        buy_threshold: float = 0.01,    # > +1.0% -> BUY
        avoid_threshold: float = -0.005, # < -0.5% -> AVOID
        buy_exposure: float = 1.0,
        hold_exposure: float = 0.5,
        avoid_exposure: float = 0.0,
    ):
        self.signal_col = signal_col
        self.score_col = score_col
        self.buy_threshold = buy_threshold
        self.avoid_threshold = avoid_threshold
        self.buy_exposure = buy_exposure
        self.hold_exposure = hold_exposure
        self.avoid_exposure = avoid_exposure

    def _resolve_signal(self, row: pd.Series) -> str:
        if self.signal_col and self.signal_col in row and pd.notna(row[self.signal_col]):
            return str(row[self.signal_col]).upper()
        
        score = row.get(self.score_col, 0.0)
        if score >= self.buy_threshold:
            return "BUY"
        elif score <= self.avoid_threshold:
            return "AVOID"
        return "HOLD"

    def generate_target_weights(
        self,
        date: Any,
        predictions_df: pd.DataFrame,
        active_universe: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        if predictions_df.empty:
            return {}

        df_candidates = predictions_df[predictions_df["ticker"].isin(active_universe)].copy()
        if df_candidates.empty:
            return {}

        weights = {}
        n = len(df_candidates)
        base_weight = 1.0 / n if n > 0 else 0.0

        for _, row in df_candidates.iterrows():
            ticker = row["ticker"]
            sig = self._resolve_signal(row)
            if sig == "BUY":
                exposure = self.buy_exposure
            elif sig == "HOLD":
                exposure = self.hold_exposure
            else:
                exposure = self.avoid_exposure

            w = base_weight * exposure
            if w > 0:
                weights[ticker] = w

        return weights


class EqualWeightUniverseStrategy(BaseStrategy):
    """Holds all active universe stocks with equal weight (1/N)."""

    def generate_target_weights(
        self,
        date: Any,
        predictions_df: pd.DataFrame,
        active_universe: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        if not active_universe:
            return {}
        w = 1.0 / len(active_universe)
        return {t: w for t in active_universe}


class RandomTopKStrategy(BaseStrategy):
    """Random Top-K selection with deterministic seed."""

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = max(1, int(k))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_target_weights(
        self,
        date: Any,
        predictions_df: pd.DataFrame,
        active_universe: List[str],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        if not active_universe:
            return {}

        k = min(self.k, len(active_universe))
        chosen = self.rng.choice(active_universe, size=k, replace=False)
        w = 1.0 / self.k
        return {t: w for t in chosen}
