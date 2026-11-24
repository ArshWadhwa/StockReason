"""
StockReason v0.7 — Standardized Feature Attribution
====================================================
Unified feature attribution engine supporting:
1. Deep learning gradient x input attribution (method: "gradient_x_input")
2. LightGBM / Tree model attribution via native pred_contrib (method: "tree_shap")
3. Tabular heuristic / correlation fallback

All attribution outputs follow a standardized schema and naming convention:
- Renamed from ambiguous legacy "shap_explainability" to clean "feature_attribution".
- Explicit method tag: "gradient_x_input" or "tree_shap".
- Segregates top supporting factors (bullish drivers) from top conflicting factors (bearish risks).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ── Canonical Feature Display Dictionary ─────────────────────────────────────
FEATURE_METADATA = {
    "rsi_14": {
        "display": "RSI (14-day)",
        "desc": "Momentum oscillator measuring speed and magnitude of recent price moves.",
        "category": "Momentum",
    },
    "macd_line_12_26": {
        "display": "MACD Line",
        "desc": "Trend-following momentum indicator showing relationship between 12 and 26 EMAs.",
        "category": "Momentum",
    },
    "macd_hist_12_26": {
        "display": "MACD Histogram",
        "desc": "Divergence between MACD and signal line; widening indicates accelerating trend.",
        "category": "Momentum",
    },
    "dist_sma_50": {
        "display": "Distance from 50-day SMA",
        "desc": "Normalized percentage distance from 50-day moving average.",
        "category": "Trend",
    },
    "dist_sma_200": {
        "display": "Distance from 200-day SMA",
        "desc": "Macro trend baseline; trading above indicates structural bull market.",
        "category": "Trend",
    },
    "volatility_20d": {
        "display": "20-day Realized Volatility",
        "desc": "Annualized rolling standard deviation of daily logarithmic returns.",
        "category": "Volatility",
    },
    "atr_14": {
        "display": "Average True Range (14)",
        "desc": "Absolute measure of market volatility across highs, lows, and closes.",
        "category": "Volatility",
    },
    "bb_pct_b_20": {
        "display": "Bollinger %B (20)",
        "desc": "Relative location of price within Bollinger Bands; >1.0 over upper band.",
        "category": "Volatility",
    },
    "adx_14": {
        "display": "ADX Trend Strength (14)",
        "desc": "Non-directional trend strength; values >25 confirm established trend.",
        "category": "Trend Strength",
    },
    "obv": {
        "display": "On-Balance Volume (OBV)",
        "desc": "Cumulative volume flow confirming directional institutional pressure.",
        "category": "Volume",
    },
    "cmf_20": {
        "display": "Chaikin Money Flow (20)",
        "desc": "Measures accumulation vs distribution volume over 20 trading days.",
        "category": "Volume",
    },
    "sentiment_mean": {
        "display": "FinBERT News Sentiment",
        "desc": "NLP sentiment polarity score extracted from financial headlines.",
        "category": "Sentiment",
    },
    "sentiment_momentum_3d": {
        "display": "3-day Sentiment Velocity",
        "desc": "Rate of change in news tone over 3 days.",
        "category": "Sentiment",
    },
    "vix_close": {
        "display": "India VIX Volatility Index",
        "desc": "Market-wide implied volatility and fear gauge.",
        "category": "Market Context",
    },
    "nifty_50_close_return": {
        "display": "NIFTY 50 Market Return",
        "desc": "Daily percentage return of benchmark index.",
        "category": "Market Context",
    },
    "return_21d": {
        "display": "1-Month Trailing Return",
        "desc": "Historical price return over 21 trading sessions.",
        "category": "Momentum",
    },
}


@dataclass
class FeatureAttributionItem:
    """Individual factor attribution record."""
    feature: str
    feature_display: str
    description: str
    importance: float                   # Unsigned importance score [0, 1]
    impact: float                       # Signed contribution to prediction
    direction: str                      # "positive" (supporting) or "negative" (conflicting)
    current_value: Optional[float]
    category: str
    method: str                         # "gradient_x_input" or "tree_shap"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_display": self.feature_display,
            "description": self.desc,
            "importance": round(self.importance, 4),
            "impact": round(self.impact, 4),
            "direction": self.direction,
            "current_value": round(self.current_value, 4) if self.current_value is not None else None,
            "category": self.category,
            "method": self.method,
        }

    @property
    def desc(self) -> str:
        return self.description


@dataclass
class FeatureAttributionResult:
    """Consolidated attribution report for a ticker forecast."""
    ticker: str
    method: str                         # "gradient_x_input" or "tree_shap"
    target_name: str                    # e.g. "Target_21D"
    attributions: List[FeatureAttributionItem]
    top_supporting_factors: List[FeatureAttributionItem]
    top_conflicting_factors: List[FeatureAttributionItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "method": self.method,
            "target_name": self.target_name,
            "attributions": [a.to_dict() for a in self.attributions],
            "top_supporting_factors": [a.to_dict() for a in self.top_supporting_factors],
            "top_conflicting_factors": [a.to_dict() for a in self.top_conflicting_factors],
        }


class FeatureAttributionEngine:
    """
    Standardized attribution computer supporting deep models (gradient x input)
    and tree models (LightGBM pred_contrib / TreeExplainer).
    """

    @staticmethod
    def compute_gradient_x_input(
        model: Any,
        x_tensor: Any,
        feature_names: List[str],
        current_values: Optional[Dict[str, float]] = None,
        ticker: str = "TICKER",
        target_idx: int = 2,            # 21D horizon is index 2
        top_k: int = 6,
    ) -> FeatureAttributionResult:
        """
        Computes input x gradient attribution for PyTorch models.
        Strictly labeled with method: "gradient_x_input".
        """
        import torch

        model.eval()
        # Clone tensor with gradient tracking
        x = x_tensor.clone().detach().requires_grad_(True)
        out = model(x)
        
        # Zero gradients and backprop from target horizon
        if x.grad is not None:
            x.grad.zero_()
            
        if out.dim() == 2:
            out[0, target_idx].backward()
        else:
            out[target_idx].backward()

        if x.grad is None:
            return FeatureAttributionResult(
                ticker=ticker,
                method="gradient_x_input",
                target_name=f"Horizon_{target_idx}",
                attributions=[],
                top_supporting_factors=[],
                top_conflicting_factors=[],
            )

        # Gradient x Input
        grad = x.grad.squeeze(0).cpu().numpy()
        inp = x.squeeze(0).cpu().detach().numpy()
        
        # If sequence (T, F), take the last time step or average across time window
        if grad.ndim == 2:
            grad_x_inp = (grad[-1] * inp[-1])
            grad_abs = np.abs(grad_x_inp)
        else:
            grad_x_inp = grad * inp
            grad_abs = np.abs(grad_x_inp)

        num_features = min(len(feature_names), len(grad_x_inp))
        tot_imp = float(np.sum(grad_abs[:num_features])) + 1e-10

        items: List[FeatureAttributionItem] = []
        for i in range(num_features):
            feat = feature_names[i]
            meta = FEATURE_METADATA.get(feat, {})
            disp = meta.get("display", feat.replace("_", " ").title())
            desc = meta.get("desc", f"Feature {feat}")
            cat = meta.get("category", "Technical")

            raw_imp = float(grad_abs[i]) / tot_imp
            signed_impact = float(grad_x_inp[i]) / tot_imp
            direction = "positive" if signed_impact >= 0 else "negative"
            curr_val = current_values.get(feat) if current_values else None

            items.append(FeatureAttributionItem(
                feature=feat,
                feature_display=disp,
                description=desc,
                importance=raw_imp,
                impact=signed_impact,
                direction=direction,
                current_value=curr_val,
                category=cat,
                method="gradient_x_input",
            ))

        # Sort by raw importance descending
        items.sort(key=lambda x: x.importance, reverse=True)

        supporting = [it for it in items if it.direction == "positive"][:top_k]
        conflicting = [it for it in items if it.direction == "negative"][:top_k]

        return FeatureAttributionResult(
            ticker=ticker,
            method="gradient_x_input",
            target_name=f"Horizon_{target_idx}",
            attributions=items[:top_k * 2],
            top_supporting_factors=supporting,
            top_conflicting_factors=conflicting,
        )

    @staticmethod
    def compute_tree_shap(
        booster: Any,
        features_row: Union[np.ndarray, pd.Series, pd.DataFrame],
        feature_names: List[str],
        current_values: Optional[Dict[str, float]] = None,
        ticker: str = "TICKER",
        target_name: str = "Return_21D",
        top_k: int = 6,
    ) -> FeatureAttributionResult:
        """
        Computes Tree SHAP attribution for LightGBM models using native pred_contrib=True.
        Does not require external `shap` package; executes directly via LightGBM C++ core.
        Strictly labeled with method: "tree_shap".
        """
        if isinstance(features_row, pd.Series):
            arr = features_row.values.reshape(1, -1)
        elif isinstance(features_row, pd.DataFrame):
            arr = features_row.values
        else:
            arr = np.asarray(features_row).reshape(1, -1)

        # LightGBM booster or sklearn LightGBM API
        if hasattr(booster, "booster_"):
            actual_booster = booster.booster_
        else:
            actual_booster = booster

        # predict with pred_contrib=True returns (N, num_features + 1), where last is bias
        contribs = actual_booster.predict(arr, pred_contrib=True)[0]
        feature_contribs = contribs[:-1]  # Exclude expected value (bias)

        num_features = min(len(feature_names), len(feature_contribs))
        tot_abs = float(np.sum(np.abs(feature_contribs[:num_features]))) + 1e-10

        items: List[FeatureAttributionItem] = []
        for i in range(num_features):
            feat = feature_names[i]
            meta = FEATURE_METADATA.get(feat, {})
            disp = meta.get("display", feat.replace("_", " ").title())
            desc = meta.get("desc", f"Feature {feat}")
            cat = meta.get("category", "Technical")

            raw_val = float(feature_contribs[i])
            imp = abs(raw_val) / tot_abs
            direction = "positive" if raw_val >= 0 else "negative"
            curr_val = current_values.get(feat) if current_values else None

            items.append(FeatureAttributionItem(
                feature=feat,
                feature_display=disp,
                description=desc,
                importance=imp,
                impact=raw_val,
                direction=direction,
                current_value=curr_val,
                category=cat,
                method="tree_shap",
            ))

        items.sort(key=lambda x: x.importance, reverse=True)
        supporting = [it for it in items if it.direction == "positive"][:top_k]
        conflicting = [it for it in items if it.direction == "negative"][:top_k]

        return FeatureAttributionResult(
            ticker=ticker,
            method="tree_shap",
            target_name=target_name,
            attributions=items[:top_k * 2],
            top_supporting_factors=supporting,
            top_conflicting_factors=conflicting,
        )

    @classmethod
    def compute_attribution(
        cls,
        model: Any,
        features: Any,
        feature_names: List[str],
        current_values: Optional[Dict[str, float]] = None,
        ticker: str = "TICKER",
        top_k: int = 6,
    ) -> FeatureAttributionResult:
        """
        Unified polymorphic dispatcher: auto-detects PyTorch vs LightGBM models.
        """
        # Detect LightGBM
        if hasattr(model, "predict") and (hasattr(model, "booster_") or hasattr(model, "num_trees")):
            return cls.compute_tree_shap(
                booster=model,
                features_row=features,
                feature_names=feature_names,
                current_values=current_values,
                ticker=ticker,
                top_k=top_k,
            )
        # Default PyTorch / Neural Network
        else:
            return cls.compute_gradient_x_input(
                model=model,
                x_tensor=features,
                feature_names=feature_names,
                current_values=current_values,
                ticker=ticker,
                top_k=top_k,
            )
