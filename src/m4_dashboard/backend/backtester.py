from typing import Dict, Any, List
import math

class BacktestEngine:
    """
    Backtesting engine evaluating trading signals against historical returns.
    Computes Sharpe ratio, maximum drawdown, win rate, and benchmarks against:
    1. Benchmark (Buy & Hold NIFTY 50)
    2. Random-strategy baseline
    """

    @staticmethod
    def calculate_metrics(returns: List[float]) -> Dict[str, Any]:
        if not returns:
            return {
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            }

        cum_return = 1.0
        peak = 1.0
        max_dd = 0.0
        wins = 0

        for r in returns:
            cum_return *= (1.0 + r)
            if cum_return > peak:
                peak = cum_return
            dd = (cum_return - peak) / peak
            if dd < max_dd:
                max_dd = dd
            if r > 0:
                wins += 1

        n = len(returns)
        mean_r = sum(returns) / n
        var_r = sum((r - mean_r) ** 2 for r in returns) / max(1, n - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-6
        
        # Annualized values (assuming daily bars, 252 days)
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
        ann_return = (cum_return ** (252 / max(1, n))) - 1.0 if cum_return > 0 else -1.0

        return {
            "cumulative_return": round((cum_return - 1.0) * 100, 2),
            "annualized_return": round(ann_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "win_rate": round((wins / n) * 100, 2),
            "total_trades": n
        }

    @staticmethod
    def simulate_strategy_vs_random(
        price_series: List[Dict[str, Any]],
        signals: List[str]
    ) -> Dict[str, Any]:
        """
        Executes strategy trades where:
        BUY -> 100% long
        HOLD -> 50% long
        AVOID -> 0% long (cash)
        """
        if len(price_series) < 2 or len(signals) < len(price_series) - 1:
            return {}

        strat_returns = []
        bench_returns = []
        rand_returns = []

        import random
        rnd = random.Random(42)

        for i in range(len(price_series) - 1):
            p0 = price_series[i]["close"]
            p1 = price_series[i + 1]["close"]
            daily_asset_ret = (p1 - p0) / p0
            bench_returns.append(daily_asset_ret)

            sig = signals[i] if i < len(signals) else "HOLD"
            weight = 1.0 if sig == "BUY" else (0.5 if sig == "HOLD" else 0.0)
            strat_returns.append(daily_asset_ret * weight)

            rand_choice = rnd.choice([1.0, 0.5, 0.0])
            rand_returns.append(daily_asset_ret * rand_choice)

        return {
            "strategy": BacktestEngine.calculate_metrics(strat_returns),
            "benchmark": BacktestEngine.calculate_metrics(bench_returns),
            "random_baseline": BacktestEngine.calculate_metrics(rand_returns)
        }
