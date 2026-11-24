"""
Comprehensive Performance and Risk Metrics for v0.7 Backtesting.
================================================================
Calculates:
- Cumulative Return (Gross & Net)
- CAGR (Gross & Net)
- Annualized Sharpe Ratio (Gross & Net)
- Annualized Sortino Ratio (Gross & Net)
- Max Drawdown (Gross & Net) & Drawdown Duration
- Calmar Ratio (Net)
- Annualized Turnover
- Hit Rate (Win Rate % on closed trades)
- Average Win / Average Loss & Profit Factor
- Exposure (Average fraction of capital invested in equities)
- Cost Drag (Gross vs Net return delta)
- Persistence of equity curves to Parquet and JSON
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    # Cumulative & Annualized Returns
    cumulative_return_gross: float
    cumulative_return_net: float
    cagr_gross: float
    cagr_net: float

    # Risk-adjusted Returns
    annualized_sharpe_gross: float
    annualized_sharpe_net: float
    annualized_sortino_gross: float
    annualized_sortino_net: float

    # Drawdowns
    max_drawdown_gross: float
    max_drawdown_net: float
    max_drawdown_duration_days: int
    calmar_ratio_net: float

    # Trading & Execution Characteristics
    annualized_turnover: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float
    exposure: float

    # Costs
    total_costs_paid: float
    cost_drag_bps: float

    # Timing
    start_date: str
    end_date: str
    trading_days: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cumulative_return_gross_pct": round(self.cumulative_return_gross * 100.0, 2),
            "cumulative_return_net_pct": round(self.cumulative_return_net * 100.0, 2),
            "cagr_gross_pct": round(self.cagr_gross * 100.0, 2),
            "cagr_net_pct": round(self.cagr_net * 100.0, 2),
            "annualized_sharpe_gross": round(self.annualized_sharpe_gross, 2),
            "annualized_sharpe_net": round(self.annualized_sharpe_net, 2),
            "annualized_sortino_gross": round(self.annualized_sortino_gross, 2),
            "annualized_sortino_net": round(self.annualized_sortino_net, 2),
            "max_drawdown_gross_pct": round(self.max_drawdown_gross * 100.0, 2),
            "max_drawdown_net_pct": round(self.max_drawdown_net * 100.0, 2),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "calmar_ratio_net": round(self.calmar_ratio_net, 2),
            "annualized_turnover": round(self.annualized_turnover, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "hit_rate_pct": round(self.hit_rate * 100.0, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "win_loss_ratio": round(self.win_loss_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "exposure_pct": round(self.exposure * 100.0, 2),
            "total_costs_paid": round(self.total_costs_paid, 2),
            "cost_drag_bps": round(self.cost_drag_bps, 1),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
        }


def compute_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Computes peak-to-date drawdown series."""
    peak = equity_series.cummax()
    return (equity_series - peak) / peak


def compute_max_drawdown_duration(equity_series: pd.Series) -> int:
    """Computes the maximum consecutive days spent below previous high watermark."""
    peak = equity_series.cummax()
    is_underwater = equity_series < peak
    
    max_duration = 0
    current_duration = 0
    for underwater in is_underwater:
        if underwater:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0
    return max_duration


def compute_sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_daily = risk_free_annual / 252.0
    excess_returns = daily_returns - rf_daily
    std = excess_returns.std(ddof=1)
    if std is None or np.isnan(std) or std <= 1e-12:
        return 0.0
    mean_excess = excess_returns.mean()
    return float(np.sqrt(252.0) * (mean_excess / std))


def compute_sortino_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_daily = risk_free_annual / 252.0
    excess_returns = daily_returns - rf_daily
    downside_returns = np.minimum(0.0, excess_returns)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std is None or np.isnan(downside_std) or downside_std <= 1e-12:
        return 0.0
    mean_excess = excess_returns.mean()
    return float(np.sqrt(252.0) * (mean_excess / downside_std))


def compute_backtest_metrics(
    daily_df: pd.DataFrame,
    trades: List[Any],
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """
    Computes all standard portfolio metrics given daily records DataFrame and list of closed trades.
    daily_df must contain: ['date', 'gross_equity', 'net_equity', 'invested_value', 'turnover', 'costs']
    """
    if daily_df.empty or len(daily_df) < 1:
        return BacktestMetrics(
            cumulative_return_gross=0.0,
            cumulative_return_net=0.0,
            cagr_gross=0.0,
            cagr_net=0.0,
            annualized_sharpe_gross=0.0,
            annualized_sharpe_net=0.0,
            annualized_sortino_gross=0.0,
            annualized_sortino_net=0.0,
            max_drawdown_gross=0.0,
            max_drawdown_net=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio_net=0.0,
            annualized_turnover=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            hit_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            exposure=0.0,
            total_costs_paid=0.0,
            cost_drag_bps=0.0,
            start_date="",
            end_date="",
            trading_days=0,
        )

    trading_days = len(daily_df)
    start_date = str(daily_df.iloc[0]["date"])
    end_date = str(daily_df.iloc[-1]["date"])

    initial_gross = float(daily_df.iloc[0]["gross_equity"])
    final_gross = float(daily_df.iloc[-1]["gross_equity"])
    initial_net = float(daily_df.iloc[0]["net_equity"])
    final_net = float(daily_df.iloc[-1]["net_equity"])

    cum_gross = (final_gross - initial_gross) / initial_gross if initial_gross > 0 else 0.0
    cum_net = (final_net - initial_net) / initial_net if initial_net > 0 else 0.0

    # CAGR calculation
    years = max(1.0 / 252.0, trading_days / 252.0)
    cagr_gross = ((final_gross / initial_gross) ** (1.0 / years) - 1.0) if (initial_gross > 0 and final_gross > 0) else -1.0
    cagr_net = ((final_net / initial_net) ** (1.0 / years) - 1.0) if (initial_net > 0 and final_net > 0) else -1.0

    # Daily returns
    gross_returns = daily_df["gross_equity"].pct_change().dropna()
    net_returns = daily_df["net_equity"].pct_change().dropna()

    sharpe_gross = compute_sharpe_ratio(gross_returns, risk_free_rate)
    sharpe_net = compute_sharpe_ratio(net_returns, risk_free_rate)

    sortino_gross = compute_sortino_ratio(gross_returns, risk_free_rate)
    sortino_net = compute_sortino_ratio(net_returns, risk_free_rate)

    # Drawdowns
    dd_gross = compute_drawdown_series(daily_df["gross_equity"])
    dd_net = compute_drawdown_series(daily_df["net_equity"])
    max_dd_gross = float(dd_gross.min()) if not dd_gross.empty else 0.0
    max_dd_net = float(dd_net.min()) if not dd_net.empty else 0.0
    max_dd_duration = compute_max_drawdown_duration(daily_df["net_equity"])

    # Calmar ratio (CAGR / |MaxDD|)
    calmar_net = (cagr_net / abs(max_dd_net)) if abs(max_dd_net) > 1e-6 else (cagr_net if cagr_net > 0 else 0.0)

    # Turnover
    total_traded_value = float(daily_df["turnover"].sum()) if "turnover" in daily_df.columns else 0.0
    avg_equity = float(daily_df["net_equity"].mean()) if not daily_df.empty else 1.0
    annualized_turnover = ((total_traded_value / 2.0) / avg_equity) * (252.0 / trading_days) if avg_equity > 0 else 0.0

    # Exposure
    if "invested_value" in daily_df.columns:
        exposure_series = daily_df["invested_value"] / daily_df["net_equity"]
        exposure = float(exposure_series.mean())
    else:
        exposure = 1.0

    # Costs paid
    total_costs = float(daily_df["costs"].sum()) if "costs" in daily_df.columns else 0.0
    cost_drag_bps = (cum_gross - cum_net) * 10000.0

    # Trade statistics
    total_trades = len(trades)
    winning_trades = 0
    losing_trades = 0
    win_pnls: List[float] = []
    loss_pnls: List[float] = []

    for t in trades:
        pnl = getattr(t, "net_pnl", 0.0)
        if pnl > 0:
            winning_trades += 1
            win_pnls.append(pnl)
        elif pnl < 0:
            losing_trades += 1
            loss_pnls.append(abs(pnl))

    hit_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (avg_win if avg_win > 0 else 0.0)
    
    total_gross_wins = sum(win_pnls)
    total_gross_losses = sum(loss_pnls)
    profit_factor = (total_gross_wins / total_gross_losses) if total_gross_losses > 0 else (100.0 if total_gross_wins > 0 else 0.0)

    return BacktestMetrics(
        cumulative_return_gross=cum_gross,
        cumulative_return_net=cum_net,
        cagr_gross=cagr_gross,
        cagr_net=cagr_net,
        annualized_sharpe_gross=sharpe_gross,
        annualized_sharpe_net=sharpe_net,
        annualized_sortino_gross=sortino_gross,
        annualized_sortino_net=sortino_net,
        max_drawdown_gross=max_dd_gross,
        max_drawdown_net=max_dd_net,
        max_drawdown_duration_days=max_dd_duration,
        calmar_ratio_net=calmar_net,
        annualized_turnover=annualized_turnover,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        exposure=exposure,
        total_costs_paid=total_costs,
        cost_drag_bps=cost_drag_bps,
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
    )


def save_equity_curve(
    daily_df: pd.DataFrame,
    parquet_path: Optional[Union[str, Path]] = None,
    json_path: Optional[Union[str, Path]] = None,
) -> None:
    """Saves the daily equity curve to Parquet and/or JSON."""
    if daily_df.empty:
        return

    # Create parent dirs if necessary
    if parquet_path:
        p = Path(parquet_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_parquet(p, index=False)

    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_json(p, orient="records", date_format="iso", indent=2)
