"""
StockReason v0.7 Cost-Aware, Honest Backtesting Engine.
======================================================
"""

from src.v07.backtest.costs import TransactionCostModel, CostBreakdown
from src.v07.backtest.orders import Position, Trade, OrderExecutor
from src.v07.backtest.strategies import (
    BaseStrategy,
    TopKStrategy,
    LongFlatStrategy,
    EqualWeightUniverseStrategy,
    RandomTopKStrategy,
)
from src.v07.backtest.metrics import (
    BacktestMetrics,
    compute_backtest_metrics,
    save_equity_curve,
)
from src.v07.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)

__all__ = [
    "TransactionCostModel",
    "CostBreakdown",
    "Position",
    "Trade",
    "OrderExecutor",
    "BaseStrategy",
    "TopKStrategy",
    "LongFlatStrategy",
    "EqualWeightUniverseStrategy",
    "RandomTopKStrategy",
    "BacktestMetrics",
    "compute_backtest_metrics",
    "save_equity_curve",
    "BacktestEngine",
    "BacktestResult",
]
