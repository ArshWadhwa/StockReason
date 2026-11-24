"""
Order Execution, Position Tracking, and Fill Models.
===================================================
Models market orders at open (t+1), limit-entry orders with intra-day bar checks,
stop-loss and take-profit triggers with priority and gap-down modeling.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from src.v07.backtest.costs import CostBreakdown, TransactionCostModel


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    side: str
    shares: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str
    total_costs: float
    entry_costs: Optional[CostBreakdown] = None
    exit_costs: Optional[CostBreakdown] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "side": self.side,
            "shares": self.shares,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct * 100, 4),
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
            "total_costs": round(self.total_costs, 2),
        }


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: Any
    cost_basis: float
    entry_costs: CostBreakdown
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    target_horizon_days: Optional[int] = None
    days_held: int = 0

    @property
    def current_value(self) -> float:
        return self.shares * self.entry_price

    def update_day(self) -> None:
        self.days_held += 1

    def check_intraday_exit(
        self,
        bar: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates stop-loss and take-profit on a daily bar (open, high, low, close).
        Rules:
        - If both stop and take profit touched: stop triggers first!
        - Gap-through-stop: if open < stop_loss_price, fills at open.
        - Normal stop: fills at stop_loss_price.
        - Gap-through-take-profit: if open > take_profit_price, fills at open.
        - Normal take-profit: fills at take_profit_price.
        """
        open_p = bar["open"]
        high_p = bar["high"]
        low_p = bar["low"]

        stop_hit = (self.stop_loss_price is not None and low_p <= self.stop_loss_price)
        take_hit = (self.take_profit_price is not None and high_p >= self.take_profit_price)

        if stop_hit and take_hit:
            # STOP TRIGGERS FIRST IF BOTH TOUCHED
            exit_price = open_p if open_p < self.stop_loss_price else self.stop_loss_price
            return {
                "exit_price": exit_price,
                "exit_reason": "STOP_LOSS",
            }
        elif stop_hit:
            exit_price = open_p if open_p < self.stop_loss_price else self.stop_loss_price
            return {
                "exit_price": exit_price,
                "exit_reason": "STOP_LOSS",
            }
        elif take_hit:
            exit_price = open_p if open_p > self.take_profit_price else self.take_profit_price
            return {
                "exit_price": exit_price,
                "exit_reason": "TAKE_PROFIT",
            }

        return None


class OrderExecutor:
    """
    Executes entry and exit orders applying limit order constraints, market fills, and cost models.
    """

    def __init__(self, cost_model: Optional[TransactionCostModel] = None):
        self.cost_model = cost_model or TransactionCostModel()

    def execute_market_entry(
        self,
        ticker: str,
        target_allocation: float,
        available_cash: float,
        date: Any,
        bar: Dict[str, float],
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        target_horizon_days: Optional[int] = None,
        median_volume: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Executes a market order at open price of date t+1.
        """
        open_price = bar["open"]
        if open_price <= 0:
            return None

        # Cash allocation for this ticker
        order_cash = min(available_cash, target_allocation)
        if order_cash <= 0:
            return None

        # Estimate shares (accounting roughly for fees)
        est_shares = order_cash / open_price
        costs = self.cost_model.compute_costs("BUY", est_shares, open_price, median_volume)
        
        # Adjust shares so total cost <= order_cash
        if order_cash > costs.total_cost:
            net_cash = order_cash - costs.total_cost
            shares = net_cash / open_price
            costs = self.cost_model.compute_costs("BUY", shares, open_price, median_volume)
        else:
            return None

        if shares <= 0:
            return None

        sl_price = open_price * (1.0 - stop_loss_pct) if stop_loss_pct else None
        tp_price = open_price * (1.0 + take_profit_pct) if take_profit_pct else None

        cost_basis = (shares * open_price) + costs.total_cost

        return Position(
            ticker=ticker,
            shares=shares,
            entry_price=open_price,
            entry_date=date,
            cost_basis=cost_basis,
            entry_costs=costs,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            target_horizon_days=target_horizon_days,
            days_held=0,
        )

    def execute_limit_entry(
        self,
        ticker: str,
        limit_price: float,
        target_allocation: float,
        available_cash: float,
        date: Any,
        bar: Dict[str, float],
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        target_horizon_days: Optional[int] = None,
        median_volume: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Executes a limit entry order on day t+1:
        - Fills ONLY if bar['low'] <= limit_price.
        - If open <= limit_price, fills at open (price improvement).
        - If open > limit_price and low <= limit_price, fills at limit_price.
        - If low > limit_price, does not fill (returns None).
        """
        open_price = bar["open"]
        low_price = bar["low"]

        if low_price > limit_price:
            return None  # Unfilled

        fill_price = open_price if open_price <= limit_price else limit_price
        if fill_price <= 0:
            return None

        order_cash = min(available_cash, target_allocation)
        if order_cash <= 0:
            return None

        est_shares = order_cash / fill_price
        costs = self.cost_model.compute_costs("BUY", est_shares, fill_price, median_volume)
        
        if order_cash > costs.total_cost:
            net_cash = order_cash - costs.total_cost
            shares = net_cash / fill_price
            costs = self.cost_model.compute_costs("BUY", shares, fill_price, median_volume)
        else:
            return None

        if shares <= 0:
            return None

        sl_price = fill_price * (1.0 - stop_loss_pct) if stop_loss_pct else None
        tp_price = fill_price * (1.0 + take_profit_pct) if take_profit_pct else None
        cost_basis = (shares * fill_price) + costs.total_cost

        return Position(
            ticker=ticker,
            shares=shares,
            entry_price=fill_price,
            entry_date=date,
            cost_basis=cost_basis,
            entry_costs=costs,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            target_horizon_days=target_horizon_days,
            days_held=0,
        )

    def execute_exit(
        self,
        position: Position,
        exit_price: float,
        exit_date: Any,
        exit_reason: str,
        median_volume: Optional[float] = None,
    ) -> Trade:
        """
        Closes an open position and computes trade P&L and net proceeds.
        """
        shares = position.shares
        exit_costs = self.cost_model.compute_costs("SELL", shares, exit_price, median_volume)
        
        gross_proceeds = shares * exit_price
        net_proceeds = gross_proceeds - exit_costs.total_cost
        
        gross_pnl = gross_proceeds - (shares * position.entry_price)
        total_costs = position.entry_costs.total_cost + exit_costs.total_cost
        net_pnl = gross_pnl - total_costs
        
        return_pct = net_pnl / position.cost_basis if position.cost_basis > 0 else 0.0

        return Trade(
            ticker=position.ticker,
            entry_date=str(position.entry_date),
            exit_date=str(exit_date),
            side="BUY",
            shares=shares,
            entry_price=position.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_days=position.days_held,
            exit_reason=exit_reason,
            total_costs=total_costs,
            entry_costs=position.entry_costs,
            exit_costs=exit_costs,
        )
