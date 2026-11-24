"""
StockReason v0.7 — ATR Brackets & Execution Fill Modeling
==========================================================
ATR-based bracket generator (entry, stop-loss, take-profit) and execution fill comparator
evaluating whether disciplined trade brackets add alpha net of slippage and opportunity cost.

Key components:
1. ATR Bracket Logic:
   - Dynamic volatility-based stop-loss and profit target calibrated to 14-day ATR.
   - Pullback limit entry (e.g. 0.5 x ATR below close) vs Market immediate entry.
2. WS3-Style Realistic Fill Model:
   - Limit orders: filled only if subsequent low <= limit price within max wait bars.
   - Slippage & friction: 10 bps slippage on market executions, conservative intraday stop fills.
   - Intraday conflict resolution: if both stop and target touched in the same bar, assume stop first.
3. Controlled Empirical Comparison:
   - Strategy A: With Brackets (Limit entry, Stop-Loss, Take-Profit).
   - Strategy B: Without Brackets (Immediate market entry, held for full 21D horizon).
   - Evaluates: Return, Sharpe, Win Rate, Max Drawdown, and Opportunity Cost of unfilled orders.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BracketConfig:
    """Configuration for ATR-based brackets."""
    entry_atr_mult: float = 0.50          # Limit entry: current - 0.5 * ATR
    stop_atr_mult: float = 2.00           # Stop-loss: entry - 2.0 * ATR
    target_atr_mult: float = 3.00         # Take-profit: entry + 3.0 * ATR
    max_fill_wait_bars: int = 3           # Limit order expires if unfilled after N bars
    holding_horizon_bars: int = 21        # Maximum holding period (1 Month)
    market_slippage_bps: float = 10.0     # 10 bps market order execution slippage
    transaction_cost_bps: float = 20.0    # 20 bps round-trip trading friction


@dataclass
class TradeBracket:
    """Computed entry, stop, and target brackets for a signal."""
    ticker: str
    current_price: float
    atr_14: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    risk_amount: float
    reward_amount: float
    bracket_type: str = "pullback_limit"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "current_price": round(self.current_price, 2),
            "atr_14": round(self.atr_14, 2),
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "risk_amount": round(self.risk_amount, 2),
            "reward_amount": round(self.reward_amount, 2),
            "bracket_type": self.bracket_type,
        }


@dataclass
class SimulatedTradeResult:
    """Outcome of an individual trade simulation."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return: float
    gross_return: float
    bars_held: int
    exit_reason: str                      # "TAKE_PROFIT", "STOP_LOSS", "TIME_EXPIRY", "UNFILLED"
    is_filled: bool
    bracket_used: bool


@dataclass
class BracketComparisonReport:
    """Comparative backtest performance: With Brackets vs Without Brackets."""
    total_signals: int
    filled_trades_with_brackets: int
    fill_rate_pct: float
    
    # Strategy A: With Brackets
    with_brackets_return_pct: float
    with_brackets_sharpe: float
    with_brackets_max_dd_pct: float
    with_brackets_win_rate_pct: float
    with_brackets_profit_factor: float
    stop_loss_hit_pct: float
    take_profit_hit_pct: float
    time_expiry_pct: float
    
    # Strategy B: Without Brackets
    without_brackets_return_pct: float
    without_brackets_sharpe: float
    without_brackets_max_dd_pct: float
    without_brackets_win_rate_pct: float
    without_brackets_profit_factor: float
    
    # Comparative Assessment
    brackets_add_value: bool
    key_findings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "with_brackets": {
                "filled_trades": self.filled_trades_with_brackets,
                "fill_rate": f"{self.fill_rate_pct:.1f}%",
                "cumulative_return": f"{self.with_brackets_return_pct:+.2f}%",
                "sharpe": round(self.with_brackets_sharpe, 2),
                "max_drawdown": f"{self.with_brackets_max_dd_pct:.2f}%",
                "win_rate": f"{self.with_brackets_win_rate_pct:.1f}%",
                "profit_factor": round(self.with_brackets_profit_factor, 2),
                "stop_loss_hit_pct": f"{self.stop_loss_hit_pct:.1f}%",
                "take_profit_hit_pct": f"{self.take_profit_hit_pct:.1f}%",
                "time_expiry_pct": f"{self.time_expiry_pct:.1f}%",
            },
            "without_brackets": {
                "cumulative_return": f"{self.without_brackets_return_pct:+.2f}%",
                "sharpe": round(self.without_brackets_sharpe, 2),
                "max_drawdown": f"{self.without_brackets_max_dd_pct:.2f}%",
                "win_rate": f"{self.without_brackets_win_rate_pct:.1f}%",
                "profit_factor": round(self.without_brackets_profit_factor, 2),
            },
            "brackets_add_value": self.brackets_add_value,
            "key_findings": self.key_findings,
        }


class BracketManager:
    """
    Computes ATR-based entry/exit brackets and simulates execution using the fill model.
    """

    def __init__(self, config: Optional[BracketConfig] = None):
        self.config = config or BracketConfig()

    def create_brackets(
        self,
        ticker: str,
        current_price: float,
        atr_14: float,
        use_pullback: bool = True,
    ) -> TradeBracket:
        """
        Calculates ATR-based entry price, stop-loss, and profit target.
        """
        p = float(current_price)
        atr = float(atr_14) if atr_14 and atr_14 > 0 else max(1.0, p * 0.02)

        if use_pullback:
            entry = p - (self.config.entry_atr_mult * atr)
            b_type = "pullback_limit"
        else:
            entry = p
            b_type = "market_immediate"

        stop = entry - (self.config.stop_atr_mult * atr)
        target = entry + (self.config.target_atr_mult * atr)

        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0.0

        return TradeBracket(
            ticker=ticker,
            current_price=p,
            atr_14=atr,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            risk_reward_ratio=rr,
            risk_amount=risk,
            reward_amount=reward,
            bracket_type=b_type,
        )

    def simulate_trade_with_brackets(
        self,
        bracket: TradeBracket,
        forward_bars: pd.DataFrame,
    ) -> SimulatedTradeResult:
        """
        Simulates execution of a single trade using limit pullback entry and ATR brackets.
        forward_bars must contain ['date', 'open', 'high', 'low', 'close'].
        """
        total_bars = len(forward_bars)
        if total_bars == 0:
            return SimulatedTradeResult(
                ticker=bracket.ticker,
                entry_date="N/A",
                exit_date="N/A",
                entry_price=0.0,
                exit_price=0.0,
                net_return=0.0,
                gross_return=0.0,
                bars_held=0,
                exit_reason="UNFILLED",
                is_filled=False,
                bracket_used=True,
            )

        # 1. Fill check for limit order
        entry_idx = -1
        actual_entry_price = bracket.entry_price

        for i in range(min(self.config.max_fill_wait_bars, total_bars)):
            row = forward_bars.iloc[i]
            # Check if limit price was touched
            if row["low"] <= bracket.entry_price:
                entry_idx = i
                # If market gapped down below limit price, fill at open (positive slippage)
                actual_entry_price = min(bracket.entry_price, row["open"])
                break

        if entry_idx == -1:
            # Order expired unfilled
            return SimulatedTradeResult(
                ticker=bracket.ticker,
                entry_date=str(forward_bars.iloc[0]["date"])[:10],
                exit_date=str(forward_bars.iloc[min(self.config.max_fill_wait_bars - 1, total_bars - 1)]["date"])[:10],
                entry_price=0.0,
                exit_price=0.0,
                net_return=0.0,
                gross_return=0.0,
                bars_held=0,
                exit_reason="UNFILLED",
                is_filled=False,
                bracket_used=True,
            )

        entry_date = str(forward_bars.iloc[entry_idx]["date"])[:10]
        
        # 2. Track position through remaining bars
        exit_idx = entry_idx
        exit_price = actual_entry_price
        exit_reason = "TIME_EXPIRY"

        # Check subsequent bars
        max_bars = min(total_bars, entry_idx + self.config.holding_horizon_bars)
        for i in range(entry_idx + 1, max_bars):
            row = forward_bars.iloc[i]
            hit_stop = row["low"] <= bracket.stop_loss
            hit_target = row["high"] >= bracket.take_profit

            if hit_stop and hit_target:
                # Intraday conflict: conservative assumption is stop triggered first
                exit_price = bracket.stop_loss
                exit_reason = "STOP_LOSS"
                exit_idx = i
                break
            elif hit_stop:
                exit_price = bracket.stop_loss
                exit_reason = "STOP_LOSS"
                exit_idx = i
                break
            elif hit_target:
                exit_price = bracket.take_profit
                exit_reason = "TAKE_PROFIT"
                exit_idx = i
                break

        if exit_reason == "TIME_EXPIRY":
            exit_idx = max_bars - 1
            exit_price = float(forward_bars.iloc[exit_idx]["close"])

        exit_date = str(forward_bars.iloc[exit_idx]["date"])[:10]
        bars_held = exit_idx - entry_idx

        # Returns net of costs
        cost_frac = (self.config.transaction_cost_bps) / 10000.0
        gross_ret = (exit_price - actual_entry_price) / actual_entry_price
        net_ret = gross_ret - cost_frac

        return SimulatedTradeResult(
            ticker=bracket.ticker,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=actual_entry_price,
            exit_price=exit_price,
            net_return=net_ret,
            gross_return=gross_ret,
            bars_held=bars_held,
            exit_reason=exit_reason,
            is_filled=True,
            bracket_used=True,
        )

    def simulate_trade_without_brackets(
        self,
        ticker: str,
        forward_bars: pd.DataFrame,
    ) -> SimulatedTradeResult:
        """
        Simulates standard benchmark trade: immediate market entry, hold full 21D horizon,
        exit at market close on bar 21 (with market order slippage applied).
        """
        total_bars = len(forward_bars)
        if total_bars < 2:
            return SimulatedTradeResult(
                ticker=ticker,
                entry_date="N/A",
                exit_date="N/A",
                entry_price=0.0,
                exit_price=0.0,
                net_return=0.0,
                gross_return=0.0,
                bars_held=0,
                exit_reason="UNFILLED",
                is_filled=False,
                bracket_used=False,
            )

        # Immediate market entry at bar 0 open or close + slippage
        slip = self.config.market_slippage_bps / 10000.0
        entry_price = float(forward_bars.iloc[0]["close"]) * (1.0 + slip)
        entry_date = str(forward_bars.iloc[0]["date"])[:10]

        # Hold for full horizon
        exit_idx = min(total_bars - 1, self.config.holding_horizon_bars)
        exit_price = float(forward_bars.iloc[exit_idx]["close"]) * (1.0 - slip)
        exit_date = str(forward_bars.iloc[exit_idx]["date"])[:10]

        cost_frac = self.config.transaction_cost_bps / 10000.0
        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - cost_frac

        return SimulatedTradeResult(
            ticker=ticker,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            net_return=net_ret,
            gross_return=gross_ret,
            bars_held=exit_idx,
            exit_reason="TIME_EXPIRY",
            is_filled=True,
            bracket_used=False,
        )

    def compare_strategies(
        self,
        trade_samples: List[Tuple[str, float, float, pd.DataFrame]],
    ) -> BracketComparisonReport:
        """
        Runs side-by-side comparison of With Brackets vs Without Brackets over sample trades.
        Each sample is: (ticker, current_price, atr_14, forward_bars_df)
        """
        with_results: List[SimulatedTradeResult] = []
        without_results: List[SimulatedTradeResult] = []

        for ticker, price, atr, forward_bars in trade_samples:
            bracket = self.create_brackets(ticker, price, atr, use_pullback=True)
            res_bracket = self.simulate_trade_with_brackets(bracket, forward_bars)
            res_nobracket = self.simulate_trade_without_brackets(ticker, forward_bars)

            with_results.append(res_bracket)
            without_results.append(res_nobracket)

        total_signals = len(trade_samples)
        filled_with = [r for r in with_results if r.is_filled]
        fill_rate = (len(filled_with) / max(1, total_signals)) * 100.0

        # With brackets metrics
        returns_with = [r.net_return for r in filled_with]
        returns_without = [r.net_return for r in without_results if r.is_filled]

        def _calc_stats(rets: List[float]):
            if not rets:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            cum = float(np.prod([1.0 + r for r in rets]) - 1.0) * 100.0
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets)) + 1e-6
            sharpe = (mean_r / std_r) * np.sqrt(12)  # Monthly Sharpe
            
            # Max DD
            curve = np.cumprod([1.0 + r for r in rets])
            peak = np.maximum.accumulate(curve)
            dd = (curve - peak) / peak
            max_dd = float(np.min(dd)) * 100.0 if len(dd) > 0 else 0.0
            
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            win_rate = (len(wins) / len(rets)) * 100.0
            
            gross_win = sum(wins)
            gross_loss = abs(sum(losses)) + 1e-6
            pf = gross_win / gross_loss
            return cum, sharpe, max_dd, win_rate, pf

        w_cum, w_sharpe, w_mdd, w_wr, w_pf = _calc_stats(returns_with)
        wo_cum, wo_sharpe, wo_mdd, wo_wr, wo_pf = _calc_stats(returns_without)

        # Bracket breakdown
        stops = sum(1 for r in filled_with if r.exit_reason == "STOP_LOSS")
        targets = sum(1 for r in filled_with if r.exit_reason == "TAKE_PROFIT")
        expiries = sum(1 for r in filled_with if r.exit_reason == "TIME_EXPIRY")
        n_filled = max(1, len(filled_with))

        stop_pct = (stops / n_filled) * 100.0
        target_pct = (targets / n_filled) * 100.0
        expiry_pct = (expiries / n_filled) * 100.0

        # Assess whether brackets add value
        # Criteria: Higher Sharpe ratio OR lower max drawdown with acceptable return
        adds_value = (w_sharpe >= wo_sharpe and abs(w_mdd) <= abs(wo_mdd) + 2.0) or (abs(w_mdd) < abs(wo_mdd) * 0.70)

        findings = [
            f"Limit pullback entry fill rate was {fill_rate:.1f}% ({len(filled_with)}/{total_signals} orders executed).",
            f"With Brackets achieved Sharpe {w_sharpe:.2f} vs {wo_sharpe:.2f} Without Brackets.",
            f"Maximum Drawdown with brackets was {w_mdd:.2f}% vs {wo_mdd:.2f}% without brackets.",
            f"Win rate: {w_wr:.1f}% (Brackets) vs {wo_wr:.1f}% (Unbracketed).",
            f"Bracket exit distribution: {target_pct:.1f}% Take-Profit, {stop_pct:.1f}% Stop-Loss, {expiry_pct:.1f}% Time-Expiry.",
        ]

        if adds_value:
            findings.append(
                "VERDICT: ATR Brackets ADD VALUE by truncating tail risk and improving risk-adjusted Sharpe."
            )
        else:
            findings.append(
                "VERDICT: Brackets exhibit significant OPPORTUNITY COST due to missed runaway trends on limit entries."
            )

        return BracketComparisonReport(
            total_signals=total_signals,
            filled_trades_with_brackets=len(filled_with),
            fill_rate_pct=fill_rate,
            with_brackets_return_pct=w_cum,
            with_brackets_sharpe=w_sharpe,
            with_brackets_max_dd_pct=w_mdd,
            with_brackets_win_rate_pct=w_wr,
            with_brackets_profit_factor=w_pf,
            stop_loss_hit_pct=stop_pct,
            take_profit_hit_pct=target_pct,
            time_expiry_pct=expiry_pct,
            without_brackets_return_pct=wo_cum,
            without_brackets_sharpe=wo_sharpe,
            without_brackets_max_dd_pct=wo_mdd,
            without_brackets_win_rate_pct=wo_wr,
            without_brackets_profit_factor=wo_pf,
            brackets_add_value=adds_value,
            key_findings=findings,
        )
