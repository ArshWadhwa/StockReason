"""
Transaction Cost & Fee Model for Indian Equity Delivery Trading.
================================================================
Models all statutory charges, brokerage, exchange fees, taxes, and liquidity-scaled slippage:
1. Brokerage: e.g. 0.03% capped at ₹20 per executed order (or custom % / flat).
2. STT (Securities Transaction Tax): 0.1% on both delivery buy and delivery sell.
3. Exchange Transaction Charges (NSE): 0.00345% on trade value.
4. SEBI Turnover Fee: 0.0001% on trade value (₹10 per crore).
5. Stamp Duty: 0.015% on buy side only.
6. GST: 18% on (Brokerage + Exchange Txn Charges + SEBI Turnover Fee).
7. Slippage: In basis points (bps), optionally scaled by trade size relative to median daily volume.
   Default conservative delivery round-trip total friction ~25-40 bps.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass
class CostBreakdown:
    side: str
    shares: float
    price: float
    trade_value: float
    brokerage: float
    stt: float
    exchange_txn_fee: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    slippage_bps: float
    slippage_cost: float
    total_cost: float
    cost_bps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "shares": self.shares,
            "price": self.price,
            "trade_value": self.trade_value,
            "brokerage": round(self.brokerage, 4),
            "stt": round(self.stt, 4),
            "exchange_txn_fee": round(self.exchange_txn_fee, 4),
            "sebi_fee": round(self.sebi_fee, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "gst": round(self.gst, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "slippage_cost": round(self.slippage_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "cost_bps": round(self.cost_bps, 2),
        }


class TransactionCostModel:
    """
    Computes precise transaction costs for Indian equity delivery trading.
    """

    def __init__(
        self,
        brokerage_pct: float = 0.0003,       # 0.03%
        brokerage_flat_cap: float = 20.0,    # Max ₹20 per order
        stt_pct: float = 0.0010,             # 0.1% on delivery buy and sell
        exchange_txn_pct: float = 0.0000345, # 0.00345%
        sebi_turnover_pct: float = 0.000001, # 0.0001%
        stamp_duty_pct: float = 0.00015,     # 0.015% on buy only
        gst_pct: float = 0.18,               # 18% on (brokerage + txn + sebi)
        base_slippage_bps: float = 5.0,      # 5 bps base slippage per side
        impact_factor: float = 0.1,          # Liquidity scaling factor
        use_statutory_charges: bool = True,
    ):
        self.brokerage_pct = max(0.0, brokerage_pct)
        self.brokerage_flat_cap = max(0.0, brokerage_flat_cap)
        self.stt_pct = max(0.0, stt_pct)
        self.exchange_txn_pct = max(0.0, exchange_txn_pct)
        self.sebi_turnover_pct = max(0.0, sebi_turnover_pct)
        self.stamp_duty_pct = max(0.0, stamp_duty_pct)
        self.gst_pct = max(0.0, gst_pct)
        self.base_slippage_bps = max(0.0, base_slippage_bps)
        self.impact_factor = max(0.0, impact_factor)
        self.use_statutory_charges = use_statutory_charges

    @classmethod
    def zero_cost(cls) -> "TransactionCostModel":
        """Returns frictionless cost model (zero fees, zero slippage)."""
        return cls(
            brokerage_pct=0.0,
            brokerage_flat_cap=0.0,
            stt_pct=0.0,
            exchange_txn_pct=0.0,
            sebi_turnover_pct=0.0,
            stamp_duty_pct=0.0,
            gst_pct=0.0,
            base_slippage_bps=0.0,
            impact_factor=0.0,
            use_statutory_charges=False,
        )

    @classmethod
    def flat_bps(cls, round_trip_bps: float = 25.0) -> "TransactionCostModel":
        """
        Creates a simplified cost model with a flat basis point fee per side (half round-trip).
        """
        half_bps = max(0.0, round_trip_bps / 2.0)
        return cls(
            brokerage_pct=0.0,
            brokerage_flat_cap=0.0,
            stt_pct=0.0,
            exchange_txn_pct=0.0,
            sebi_turnover_pct=0.0,
            stamp_duty_pct=0.0,
            gst_pct=0.0,
            base_slippage_bps=half_bps,
            impact_factor=0.0,
            use_statutory_charges=False,
        )

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None) -> "TransactionCostModel":
        if not cfg:
            return cls()
        return cls(
            brokerage_pct=float(cfg.get("brokerage_pct", 0.0003)),
            brokerage_flat_cap=float(cfg.get("brokerage_flat_cap", 20.0)),
            stt_pct=float(cfg.get("stt_pct", 0.0010)),
            exchange_txn_pct=float(cfg.get("exchange_txn_pct", 0.0000345)),
            sebi_turnover_pct=float(cfg.get("sebi_turnover_pct", 0.000001)),
            stamp_duty_pct=float(cfg.get("stamp_duty_pct", 0.00015)),
            gst_pct=float(cfg.get("gst_pct", 0.18)),
            base_slippage_bps=float(cfg.get("base_slippage_bps", 5.0)),
            impact_factor=float(cfg.get("impact_factor", 0.1)),
            use_statutory_charges=bool(cfg.get("use_statutory_charges", True)),
        )

    def calculate_slippage_bps(
        self,
        shares: float,
        median_volume: Optional[float] = None
    ) -> float:
        """
        Computes slippage in bps, scaling by square root of participation rate if median volume available.
        """
        if self.base_slippage_bps == 0.0 and self.impact_factor == 0.0:
            return 0.0
        
        slippage = self.base_slippage_bps
        if median_volume is not None and median_volume > 0 and shares > 0 and self.impact_factor > 0:
            participation = shares / median_volume
            impact_bps = self.impact_factor * math.sqrt(participation) * 10000.0
            slippage += min(impact_bps, 100.0)  # cap impact at 100 bps (1%)
        return slippage

    def compute_costs(
        self,
        side: str,
        shares: float,
        price: float,
        median_volume: Optional[float] = None,
    ) -> CostBreakdown:
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")

        shares = abs(float(shares))
        price = abs(float(price))
        trade_value = shares * price

        if trade_value == 0.0:
            return CostBreakdown(
                side=side_norm,
                shares=0.0,
                price=price,
                trade_value=0.0,
                brokerage=0.0,
                stt=0.0,
                exchange_txn_fee=0.0,
                sebi_fee=0.0,
                stamp_duty=0.0,
                gst=0.0,
                slippage_bps=0.0,
                slippage_cost=0.0,
                total_cost=0.0,
                cost_bps=0.0,
            )

        if not self.use_statutory_charges and self.base_slippage_bps == 0.0 and self.impact_factor == 0.0:
            return CostBreakdown(
                side=side_norm,
                shares=shares,
                price=price,
                trade_value=trade_value,
                brokerage=0.0,
                stt=0.0,
                exchange_txn_fee=0.0,
                sebi_fee=0.0,
                stamp_duty=0.0,
                gst=0.0,
                slippage_bps=0.0,
                slippage_cost=0.0,
                total_cost=0.0,
                cost_bps=0.0,
            )

        # 1. Slippage
        slip_bps = self.calculate_slippage_bps(shares, median_volume)
        slippage_cost = trade_value * (slip_bps / 10000.0)

        # 2. Statutory and Brokerage charges
        if self.use_statutory_charges:
            # Brokerage (min of pct and flat cap if flat cap > 0)
            raw_brokerage = trade_value * self.brokerage_pct
            if self.brokerage_flat_cap > 0:
                brokerage = min(raw_brokerage, self.brokerage_flat_cap)
            else:
                brokerage = raw_brokerage

            # STT on delivery (both buy and sell)
            stt = trade_value * self.stt_pct

            # Exchange turnover charge
            exchange_txn_fee = trade_value * self.exchange_txn_pct

            # SEBI turnover fee
            sebi_fee = trade_value * self.sebi_turnover_pct

            # Stamp duty (buy only)
            stamp_duty = (trade_value * self.stamp_duty_pct) if side_norm == "BUY" else 0.0

            # GST on (brokerage + exchange fee + sebi fee)
            gst = (brokerage + exchange_txn_fee + sebi_fee) * self.gst_pct
        else:
            brokerage = 0.0
            stt = 0.0
            exchange_txn_fee = 0.0
            sebi_fee = 0.0
            stamp_duty = 0.0
            gst = 0.0

        total_cost = brokerage + stt + exchange_txn_fee + sebi_fee + stamp_duty + gst + slippage_cost
        cost_bps = (total_cost / trade_value) * 10000.0 if trade_value > 0 else 0.0

        return CostBreakdown(
            side=side_norm,
            shares=shares,
            price=price,
            trade_value=trade_value,
            brokerage=brokerage,
            stt=stt,
            exchange_txn_fee=exchange_txn_fee,
            sebi_fee=sebi_fee,
            stamp_duty=stamp_duty,
            gst=gst,
            slippage_bps=slip_bps,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            cost_bps=cost_bps,
        )
