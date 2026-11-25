"""Statistical Benchmarks & No-Change Baseline Suite."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class NoChangeRandomWalkBaseline:
    """
    Zero-return Random Walk benchmark:
    Predicts next price = current price (hat{P}_{t+H} = P_t, hat{R} = 0.0).
    """
    def __init__(self, horizon: int = 21):
        self.horizon = horizon

    def predict(self, df: pd.DataFrame, close_col: str = "close") -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_returns, predicted_prices)."""
        close = df[close_col].values
        pred_ret = np.zeros_like(close, dtype=np.float64)
        pred_price = close * 1.0
        return pred_ret, pred_price


class HistoricalMeanBaseline:
    """Historical mean return benchmark fitted strictly on training partition."""
    def __init__(self, horizon: int = 21):
        self.horizon = horizon
        self.mean_return: float = 0.0

    def fit(self, train_df: pd.DataFrame, ret_col: str = "target_return_nominal_21d") -> "HistoricalMeanBaseline":
        vals = train_df[ret_col].dropna().values
        self.mean_return = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return self

    def predict(self, df: pd.DataFrame, close_col: str = "close") -> Tuple[np.ndarray, np.ndarray]:
        close = df[close_col].values
        pred_ret = np.full_like(close, self.mean_return, dtype=np.float64)
        pred_price = close * (1.0 + self.mean_return)
        return pred_ret, pred_price


class HistoricalMedianBaseline:
    """Historical median return benchmark fitted strictly on training partition."""
    def __init__(self, horizon: int = 21):
        self.horizon = horizon
        self.median_return: float = 0.0

    def fit(self, train_df: pd.DataFrame, ret_col: str = "target_return_nominal_21d") -> "HistoricalMedianBaseline":
        vals = train_df[ret_col].dropna().values
        self.median_return = float(np.median(vals)) if len(vals) > 0 else 0.0
        return self

    def predict(self, df: pd.DataFrame, close_col: str = "close") -> Tuple[np.ndarray, np.ndarray]:
        close = df[close_col].values
        pred_ret = np.full_like(close, self.median_return, dtype=np.float64)
        pred_price = close * (1.0 + self.median_return)
        return pred_ret, pred_price


class HierarchicalShrinkageBaseline:
    """
    Stock-level median return shrunk toward market-wide median return:
    R_shrunk = (1 - alpha) * R_stock + alpha * R_market
    """
    def __init__(self, horizon: int = 21, alpha: float = 0.5):
        self.horizon = horizon
        self.alpha = alpha
        self.market_median: float = 0.0
        self.stock_medians: Dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        ret_col: str = "target_return_nominal_21d",
        symbol_col: str = "symbol",
    ) -> "HierarchicalShrinkageBaseline":
        vals = train_df[ret_col].dropna().values
        self.market_median = float(np.median(vals)) if len(vals) > 0 else 0.0
        self.stock_medians = train_df.groupby(symbol_col)[ret_col].median().to_dict()
        return self

    def tune_alpha(
        self,
        cal_df: pd.DataFrame,
        ret_col: str = "target_return_nominal_21d",
        target_price_col: str = "target_price_21d",
        close_col: str = "close",
        symbol_col: str = "symbol",
        grid: Optional[List[float]] = None,
    ) -> float:
        """Tuning alpha on validation/calibration fold to minimize MAPE."""
        if grid is None:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        syms = cal_df[symbol_col].values
        close = cal_df[close_col].values
        y_true = cal_df[target_price_col].values

        best_a = self.alpha
        best_mape = float("inf")

        for a in grid:
            shrunk = np.array([
                (1.0 - a) * self.stock_medians.get(s, self.market_median) + a * self.market_median
                for s in syms
            ], dtype=np.float64)
            pred_p = close * (1.0 + shrunk)
            pe = np.abs(y_true - pred_p) / np.maximum(y_true, 0.01)
            mape = float(np.mean(pe) * 100.0)
            if mape < best_mape:
                best_mape = mape
                best_a = a

        self.alpha = best_a
        return self.alpha

    def predict(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        symbol_col: str = "symbol",
    ) -> Tuple[np.ndarray, np.ndarray]:
        close = df[close_col].values
        syms = df[symbol_col].values
        pred_ret = np.array([
            (1.0 - self.alpha) * self.stock_medians.get(s, self.market_median) + self.alpha * self.market_median
            for s in syms
        ], dtype=np.float64)
        pred_price = close * (1.0 + pred_ret)
        return pred_ret, pred_price
