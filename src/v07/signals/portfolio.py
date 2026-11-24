"""
StockReason v0.7 — Portfolio Rules & Position Sizing
=====================================================
Implements institutional portfolio construction rules, volatility-scaled allocation,
volatility targeting, and optional Fractional-Kelly sizing.

Portfolio Rules:
1. Sector Cap: Max 30% per sector (sum of weights in any single sector <= 0.30).
2. Sector Count Cap: Max 3 BUY positions from the same sector.
3. Single-Name Cap: Max 10% per individual asset (weight <= 0.10).
4. Volatility-Scaled Weights: Inverse-volatility weighting based on 20-day realized volatility.
5. Volatility-Targeting Scalar: Down-scales portfolio exposure if annualized portfolio volatility
   exceeds a pre-configured target (e.g. 15% target).
6. Fractional-Kelly Sizer: Available behind config flag (use_kelly=False by default),
   with explicit documentation on calibrated probability prerequisites.
7. Strictly Long-Only: Weights sum to <= 1.0, individual weights >= 0.0.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


# Default sector map for NIFTY 50 and NIFTY BANK pilot universe
DEFAULT_SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "LT.NS": "Infrastructure",
    "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    "BANKBARODA.NS": "Banking",
    "PNB.NS": "Banking",
    "AUBANK.NS": "Banking",
    "IDFCFIRSTB.NS": "Banking",
}


@dataclass
class PortfolioConfig:
    """Institutional portfolio construction configuration."""
    max_sector_weight: float = 0.30        # Max 30% total exposure per sector
    max_buys_per_sector: int = 3           # Max 3 BUY positions per sector
    max_single_weight: float = 0.10        # Max 10% per individual ticker
    min_single_weight: float = 0.01        # Minimum non-zero position size (1%)
    target_annualized_vol: float = 0.15    # 15% target portfolio volatility
    average_asset_correlation: float = 0.40 # Assumed cross-sectional equity correlation
    use_volatility_targeting: bool = True   # Enable portfolio vol targeting
    
    # Fractional Kelly Configuration
    use_kelly: bool = False                # Off by default; requires calibrated probabilities
    kelly_fraction: float = 0.25           # Quarter-Kelly (0.25x) for risk safety
    default_win_loss_ratio: float = 1.5    # Payoff ratio b if win/loss sizes not estimated
    
    # Default annualized volatility fallback if 20-day vol is missing
    fallback_volatility: float = 0.25


@dataclass
class PositionAllocation:
    """Individual asset allocation metadata."""
    ticker: str
    sector: str
    signal: str
    raw_weight: float
    capped_weight: float
    vol_targeted_weight: float
    realized_vol_20d: float
    calibrated_probability: float
    expected_net_return: float
    signal_score: float


@dataclass
class PortfolioAllocationResult:
    """Complete portfolio allocation summary."""
    weights: Dict[str, float]                      # Ticker -> final weight
    sector_weights: Dict[str, float]               # Sector -> total weight
    sector_counts: Dict[str, int]                  # Sector -> count of BUY positions
    total_equity_weight: float                     # Sum of all weights (<= 1.0)
    cash_weight: float                             # Remainder (1.0 - total_equity_weight)
    portfolio_realized_vol: float                  # Estimated annualized portfolio vol
    volatility_scalar: float                       # Applied vol target scalar (<= 1.0)
    positions: List[PositionAllocation]
    is_sector_cap_satisfied: bool
    is_sector_count_satisfied: bool
    is_single_name_cap_satisfied: bool
    is_weight_sum_satisfied: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_equity_weight": round(self.total_equity_weight, 4),
            "cash_weight": round(self.cash_weight, 4),
            "portfolio_realized_vol": round(self.portfolio_realized_vol, 4),
            "volatility_scalar": round(self.volatility_scalar, 4),
            "is_valid": (
                self.is_sector_cap_satisfied and
                self.is_sector_count_satisfied and
                self.is_single_name_cap_satisfied and
                self.is_weight_sum_satisfied
            ),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "sector_weights": {k: round(v, 4) for k, v in self.sector_weights.items()},
            "sector_counts": self.sector_counts,
            "notes": self.notes,
        }


class PortfolioSizer:
    """
    Constructs risk-managed, institutional equity portfolios from trading signals.
    Enforces sector caps, position caps, volatility weighting, and volatility targeting.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def compute_fractional_kelly_weight(
        self,
        calibrated_probability: float,
        expected_net_return: float,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> float:
        """
        Computes the continuous Fractional-Kelly allocation weight.

        MATHEMATICAL DERIVATION & PREREQUISITE WARNING:
        ----------------------------------------------
        The continuous Kelly Criterion derives from maximizing the expected log wealth:
            max_f E[log(1 + f * R)]
        For a binomial bet with win probability p and odds b = (win_amount / loss_amount),
        the full Kelly fraction is:
            f* = (p * b - (1 - p)) / b = p - (1 - p) / b

        CRITICAL REQUIREMENT FOR CALIBRATED PROBABILITIES:
        -------------------------------------------------
        Kelly sizing is hyper-sensitive to estimation error in p. If an uncalibrated model
        outputs p = 0.85 when the true conditional probability is 0.55, full Kelly sizing
        mandates an aggressive 80%+ position, almost certainly triggering catastrophic drawdown
        or margin call liquidation.
        
        Therefore:
        1. use_kelly is strictly FALSE by default.
        2. When enabled, fractional scaling (kelly_fraction = 0.25, quarter-Kelly) is enforced
           to protect against parameter estimation variance.
        3. Input p MUST be calibrated out-of-sample (via isotonic regression or Platt scaling).
        """
        p = float(calibrated_probability)
        if p <= 0.50:
            return 0.0

        if stop_loss_pct and take_profit_pct and stop_loss_pct > 0:
            b = abs(take_profit_pct) / abs(stop_loss_pct)
        else:
            b = self.config.default_win_loss_ratio

        b = max(0.5, b)
        q = 1.0 - p
        full_kelly = (p * b - q) / b

        if full_kelly <= 0.0:
            return 0.0

        fractional_kelly = full_kelly * self.config.kelly_fraction
        return float(min(1.0, max(0.0, fractional_kelly)))

    def allocate(
        self,
        signals: List[Any],
        sector_map: Optional[Dict[str, str]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
    ) -> PortfolioAllocationResult:
        """
        Main allocation routine.
        Filters BUY signals, applies sector count limits, computes volatility/Kelly weights,
        enforces sector & single-name caps, and applies volatility targeting.

        Parameters:
            signals: List of SignalResult objects or dicts representing asset evaluations.
            sector_map: Optional mapping of ticker -> sector. Defaults to DEFAULT_SECTOR_MAP.
            covariance_matrix: Optional N x N covariance matrix for exact portfolio vol.
        """
        s_map = sector_map or DEFAULT_SECTOR_MAP
        notes: List[str] = []

        # 1. Filter only actionable BUY signals
        buy_candidates = []
        for s in signals:
            if hasattr(s, "signal"):
                sig = s.signal
                ticker = s.ticker
                prob = getattr(s, "calibrated_probability", 0.55)
                exp_ret = getattr(s, "expected_net_return", 0.02)
                score = getattr(s, "signal_score", 70.0)
                vol = s.metadata.get("volatility_20d") if hasattr(s, "metadata") else None
            elif isinstance(s, dict):
                sig = s.get("signal", "HOLD")
                ticker = s.get("ticker", "UNKNOWN")
                prob = s.get("calibrated_probability", 0.55)
                exp_ret = s.get("expected_net_return", 0.02)
                score = s.get("signal_score", 70.0)
                vol = s.get("volatility_20d") or s.get("metadata", {}).get("volatility_20d")
            else:
                continue

            if sig == "BUY":
                sec = s_map.get(ticker, "Unknown")
                vol_clean = float(vol) if vol is not None and vol > 0.001 else self.config.fallback_volatility
                buy_candidates.append({
                    "ticker": ticker,
                    "sector": sec,
                    "signal": "BUY",
                    "calibrated_probability": prob,
                    "expected_net_return": exp_ret,
                    "signal_score": score,
                    "volatility_20d": vol_clean,
                })

        if not buy_candidates:
            notes.append("No BUY signals detected; 100% allocated to Cash.")
            return PortfolioAllocationResult(
                weights={},
                sector_weights={},
                sector_counts={},
                total_equity_weight=0.0,
                cash_weight=1.0,
                portfolio_realized_vol=0.0,
                volatility_scalar=1.0,
                positions=[],
                is_sector_cap_satisfied=True,
                is_sector_count_satisfied=True,
                is_single_name_cap_satisfied=True,
                is_weight_sum_satisfied=True,
                notes=notes,
            )

        # 2. Sector Count Cap: Max 3 BUYs from same sector
        # Group by sector and rank by conviction score (or expected net return)
        sector_groups: Dict[str, List[Dict[str, Any]]] = {}
        for cand in buy_candidates:
            sec = cand["sector"]
            sector_groups.setdefault(sec, []).append(cand)

        selected_candidates = []
        sector_counts: Dict[str, int] = {}
        for sec, items in sector_groups.items():
            # Sort by signal score descending, then by expected return
            sorted_items = sorted(items, key=lambda x: (x["signal_score"], x["expected_net_return"]), reverse=True)
            kept = sorted_items[:self.config.max_buys_per_sector]
            selected_candidates.extend(kept)
            sector_counts[sec] = len(kept)
            if len(sorted_items) > self.config.max_buys_per_sector:
                excluded = [x["ticker"] for x in sorted_items[self.config.max_buys_per_sector:]]
                notes.append(
                    f"Sector '{sec}' exceeded max {self.config.max_buys_per_sector} BUYs; "
                    f"pruned {excluded} to maintain sector diversification."
                )

        n_assets = len(selected_candidates)
        if n_assets == 0:
            return PortfolioAllocationResult(
                weights={},
                sector_weights={},
                sector_counts={},
                total_equity_weight=0.0,
                cash_weight=1.0,
                portfolio_realized_vol=0.0,
                volatility_scalar=1.0,
                positions=[],
                is_sector_cap_satisfied=True,
                is_sector_count_satisfied=True,
                is_single_name_cap_satisfied=True,
                is_weight_sum_satisfied=True,
                notes=notes,
            )

        # 3. Compute Raw Weights
        raw_weights = {}
        if self.config.use_kelly:
            notes.append(f"Using Fractional-Kelly sizer (fraction={self.config.kelly_fraction:.2f}).")
            for c in selected_candidates:
                kw = self.compute_fractional_kelly_weight(
                    calibrated_probability=c["calibrated_probability"],
                    expected_net_return=c["expected_net_return"],
                )
                # Volatility scaling modifier
                vol_mod = (self.config.fallback_volatility / c["volatility_20d"])
                raw_weights[c["ticker"]] = kw * vol_mod
        else:
            # Volatility-scaled weights: inverse-volatility weighting w_i ~ 1 / sigma_i
            notes.append("Using Inverse-Volatility weighting (20-day realized volatility).")
            inv_vols = {c["ticker"]: 1.0 / c["volatility_20d"] for c in selected_candidates}
            total_inv_vol = sum(inv_vols.values())
            raw_weights = {k: v / total_inv_vol for k, v in inv_vols.items()}

        # 4. Enforce Single-Name Cap (<= 10%) and Sector Cap (<= 30%)
        # Iterative waterfall constraint projection
        capped_weights = self._enforce_caps(
            weights=raw_weights,
            candidates=selected_candidates,
            max_single=self.config.max_single_weight,
            max_sector=self.config.max_sector_weight,
        )

        # 5. Volatility Targeting
        # Compute portfolio realized volatility
        # sigma_port = sqrt(sum_i w_i^2 sigma_i^2 + sum_{i!=j} w_i w_j rho sigma_i sigma_j)
        tickers = [c["ticker"] for c in selected_candidates]
        w_vec = np.array([capped_weights.get(t, 0.0) for t in tickers])
        vols_vec = np.array([c["volatility_20d"] for c in selected_candidates])

        if covariance_matrix is not None and covariance_matrix.shape == (n_assets, n_assets):
            port_var = float(w_vec.T @ covariance_matrix @ w_vec)
        else:
            # Standard analytical portfolio variance with average correlation
            rho = self.config.average_asset_correlation
            # Sigma_ij = rho * sigma_i * sigma_j for i!=j, sigma_i^2 for i==j
            cov = rho * np.outer(vols_vec, vols_vec)
            np.fill_diagonal(cov, vols_vec ** 2)
            port_var = float(w_vec.T @ cov @ w_vec)

        port_vol = float(np.sqrt(max(1e-8, port_var)))

        # Volatility scalar
        if self.config.use_volatility_targeting and port_vol > self.config.target_annualized_vol:
            vol_scalar = float(self.config.target_annualized_vol / port_vol)
            notes.append(
                f"Portfolio volatility ({port_vol*100:.1f}%) exceeds target ({self.config.target_annualized_vol*100:.1f}%). "
                f"Applying volatility scalar: {vol_scalar:.3f}."
            )
        else:
            vol_scalar = 1.0

        # Apply volatility scalar to weights
        final_weights = {t: capped_weights[t] * vol_scalar for t in tickers}
        total_equity = float(sum(final_weights.values()))
        cash_weight = float(max(0.0, 1.0 - total_equity))

        # Sector totals
        final_sector_weights: Dict[str, float] = {}
        for c in selected_candidates:
            sec = c["sector"]
            final_sector_weights[sec] = final_sector_weights.get(sec, 0.0) + final_weights[c["ticker"]]

        # Verification flags
        sector_cap_ok = all(v <= self.config.max_sector_weight + 1e-5 for v in final_sector_weights.values())
        sector_count_ok = all(c <= self.config.max_buys_per_sector for c in sector_counts.values())
        single_cap_ok = all(v <= self.config.max_single_weight + 1e-5 for v in final_weights.values())
        weight_sum_ok = total_equity <= 1.0 + 1e-5 and all(v >= -1e-6 for v in final_weights.values())

        # Construct position objects
        positions = []
        for c in selected_candidates:
            t = c["ticker"]
            pos = PositionAllocation(
                ticker=t,
                sector=c["sector"],
                signal="BUY",
                raw_weight=raw_weights[t],
                capped_weight=capped_weights[t],
                vol_targeted_weight=final_weights[t],
                realized_vol_20d=c["volatility_20d"],
                calibrated_probability=c["calibrated_probability"],
                expected_net_return=c["expected_net_return"],
                signal_score=c["signal_score"],
            )
            positions.append(pos)

        return PortfolioAllocationResult(
            weights=final_weights,
            sector_weights=final_sector_weights,
            sector_counts=sector_counts,
            total_equity_weight=total_equity,
            cash_weight=cash_weight,
            portfolio_realized_vol=port_vol,
            volatility_scalar=vol_scalar,
            positions=positions,
            is_sector_cap_satisfied=sector_cap_ok,
            is_sector_count_satisfied=sector_count_ok,
            is_single_name_cap_satisfied=single_cap_ok,
            is_weight_sum_satisfied=weight_sum_ok,
            notes=notes,
        )

    def _enforce_caps(
        self,
        weights: Dict[str, float],
        candidates: List[Dict[str, Any]],
        max_single: float,
        max_sector: float,
        max_iterations: int = 20,
    ) -> Dict[str, float]:
        """
        Iteratively projects weights onto the convex intersection of:
          1. 0 <= w_i <= max_single
          2. sum_{i in S_k} w_i <= max_sector
          3. sum_i w_i <= 1.0
        """
        w = {k: max(0.0, float(v)) for k, v in weights.items()}
        # Normalize to sum at most 1.0
        tot = sum(w.values())
        if tot > 1.0:
            w = {k: v / tot for k, v in w.items()}

        candidate_sectors = {c["ticker"]: c["sector"] for c in candidates}

        for _ in range(max_iterations):
            changed = False
            # 1. Enforce single name cap
            for t in list(w.keys()):
                if w[t] > max_single:
                    w[t] = max_single
                    changed = True

            # 2. Enforce sector cap
            sector_totals: Dict[str, float] = {}
            sector_members: Dict[str, List[str]] = {}
            for t, val in w.items():
                sec = candidate_sectors.get(t, "Unknown")
                sector_totals[sec] = sector_totals.get(sec, 0.0) + val
                sector_members.setdefault(sec, []).append(t)

            for sec, tot_val in sector_totals.items():
                if tot_val > max_sector + 1e-6:
                    scale = max_sector / tot_val
                    for t in sector_members[sec]:
                        w[t] *= scale
                    changed = True

            # 3. Enforce overall budget <= 1.0
            overall = sum(w.values())
            if overall > 1.0 + 1e-6:
                scale = 1.0 / overall
                for t in w:
                    w[t] *= scale
                changed = True

            if not changed:
                break

        return w
