"""Directional probability derivation and calibration for StockReason v0.7.

Derives P(excess return > 0) and P(net return after costs > 0), and calibrates
probabilities on out-of-fold predictions using Isotonic Regression (N >= 1000)
or Platt Scaling (N < 1000). Computes Brier score, Log Loss, Expected Calibration Error (ECE),
and generates reliability diagrams.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def derive_directional_probabilities(
    predicted_return: Union[float, np.ndarray],
    uncertainty_std: Optional[Union[float, np.ndarray]] = None,
    benchmark_return: Union[float, np.ndarray] = 0.0,
    cost_bps: float = 15.0,
    default_std: float = 0.02,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Derive uncalibrated directional probabilities assuming a normal distribution
    around the predicted return N(predicted_return, uncertainty_std^2).

    Parameters
    ----------
    predicted_return : float or np.ndarray
        Model forecast return for the horizon (e.g. 0.015 for +1.5%).
    uncertainty_std : float or np.ndarray, optional
        Forecast standard deviation / uncertainty scale (e.g. residual std or MC-dropout std).
        Must be positive. If None or <= 0, default_std is used.
    benchmark_return : float or np.ndarray, default=0.0
        Forecast or realized benchmark return (e.g. NIFTY 50 return) for excess calculation.
    cost_bps : float, default=15.0
        Round-trip transaction friction / cost in basis points (1 bp = 0.0001 = 0.01%).
    default_std : float, default=0.02
        Fallback standard deviation if uncertainty_std is missing or non-positive.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - 'prob_up': P(return > 0)
        - 'prob_excess_up': P(return - benchmark_return > 0)
        - 'prob_net_up': P(return - cost > 0)
        - 'prob_excess_net_up': P(return - benchmark_return - cost > 0)
    """
    is_scalar = np.isscalar(predicted_return)
    pred_ret = np.asarray(predicted_return, dtype=float)
    bench_ret = np.asarray(benchmark_return, dtype=float)
    cost = float(cost_bps) * 1e-4

    if uncertainty_std is None:
        std = np.full_like(pred_ret, fill_value=default_std)
    else:
        std = np.asarray(uncertainty_std, dtype=float)
        std = np.where(np.isnan(std) | (std <= 1e-8), default_std, std)

    def _prob_exceeds(mean: np.ndarray, thresh: Union[float, np.ndarray], scale: np.ndarray) -> np.ndarray:
        # P(Y > thresh) = 1 - Phi((thresh - mean) / scale) = Phi((mean - thresh) / scale)
        z = (mean - thresh) / scale
        probs = norm.cdf(z)
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    prob_up = _prob_exceeds(pred_ret, 0.0, std)
    prob_excess = _prob_exceeds(pred_ret - bench_ret, 0.0, std)
    prob_net = _prob_exceeds(pred_ret, cost, std)
    prob_excess_net = _prob_exceeds(pred_ret - bench_ret, cost, std)

    if is_scalar:
        return {
            "prob_up": float(prob_up.item()),
            "prob_excess_up": float(prob_excess.item()),
            "prob_net_up": float(prob_net.item()),
            "prob_excess_net_up": float(prob_excess_net.item()),
        }

    return {
        "prob_up": prob_up,
        "prob_excess_up": prob_excess,
        "prob_net_up": prob_net,
        "prob_excess_net_up": prob_excess_net,
    }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute mean squared error of probability forecasts (Brier score).
    Lower is better. Range: [0, 1].

    Brier = (1 / N) * sum((y_true - y_prob)^2)
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    if len(y_t) != len(y_p):
        raise ValueError(f"Lengths differ: y_true={len(y_t)}, y_prob={len(y_p)}")
    if len(y_t) == 0:
        return 0.0
    return float(np.mean((y_t - y_p) ** 2))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute binary cross-entropy (log loss).
    Lower is better.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1.0 - eps)
    if len(y_t) != len(y_p):
        raise ValueError(f"Lengths differ: y_true={len(y_t)}, y_prob={len(y_p)}")
    if len(y_t) == 0:
        return 0.0
    loss = -np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(loss)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth in {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    n_bins : int, default=10
        Number of confidence bins.
    strategy : str, default='uniform'
        'uniform' (equal bin widths) or 'quantile' (equal sample counts).

    Returns
    -------
    Dict[str, Any]
        Dictionary with ece, mce, bin_accuracies, bin_confidences, bin_counts, bin_edges.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
            "bin_edges": [],
        }

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_p, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            idx = (y_p >= low) & (y_p <= high)
        else:
            idx = (y_p >= low) & (y_p < high)

        count = int(np.sum(idx))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(y_t[idx]))
            conf = float(np.mean(y_p[idx]))
            diff = abs(acc - conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
        else:
            bin_accuracies.append(float((low + high) / 2.0))
            bin_confidences.append(float((low + high) / 2.0))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": [float(b) for b in bin_edges],
    }


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute reliability curve statistics for plotting and reporting.
    """
    cal_info = compute_ece(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = compute_brier_score(y_true, y_prob)
    loss = compute_log_loss(y_true, y_prob)

    bin_edges = cal_info["bin_edges"]
    bin_centers = [
        float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
        for i in range(len(bin_edges) - 1)
    ]

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": cal_info["bin_accuracies"],
        "bin_confidences": cal_info["bin_confidences"],
        "bin_counts": cal_info["bin_counts"],
        "bin_edges": bin_edges,
        "ece": cal_info["ece"],
        "mce": cal_info["mce"],
        "brier_score": brier,
        "log_loss": loss,
    }


def generate_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[Union[str, Path]] = None,
    y_prob_uncalibrated: Optional[np.ndarray] = None,
    label: str = "Calibrated",
    label_uncalibrated: str = "Uncalibrated",
) -> Dict[str, Any]:
    """
    Generate reliability diagram (calibration curve and confidence histogram).
    Optionally plots uncalibrated curve alongside for comparison.
    Saves figure to save_path if provided, and returns curve metadata.
    """
    import matplotlib.pyplot as plt

    curve_cal = compute_reliability_curve(y_true, y_prob, n_bins=n_bins)
    curve_uncal = (
        compute_reliability_curve(y_true, y_prob_uncalibrated, n_bins=n_bins)
        if y_prob_uncalibrated is not None
        else None
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # 1. Top plot: Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    if curve_uncal is not None:
        ax1.plot(
            curve_uncal["bin_confidences"],
            curve_uncal["bin_accuracies"],
            "s--",
            color="#d9534f",
            label=f"{label_uncalibrated} (ECE={curve_uncal['ece']:.3f}, Brier={curve_uncal['brier_score']:.3f})",
            linewidth=1.5,
            markersize=5,
        )

    ax1.plot(
        curve_cal["bin_confidences"],
        curve_cal["bin_accuracies"],
        "o-",
        color="#2b7bba",
        label=f"{label} (ECE={curve_cal['ece']:.3f}, Brier={curve_cal['brier_score']:.3f})",
        linewidth=2,
        markersize=6,
    )

    ax1.set_ylabel("Empirical Accuracy / True Frequency", fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Bottom plot: Sample counts / Sharpness histogram
    if curve_uncal is not None and y_prob_uncalibrated is not None:
        ax2.hist(
            y_prob_uncalibrated,
            bins=n_bins,
            range=(0, 1),
            color="#d9534f",
            alpha=0.4,
            label=label_uncalibrated,
            edgecolor="white",
        )
    ax2.hist(
        y_prob,
        bins=n_bins,
        range=(0, 1),
        color="#2b7bba",
        alpha=0.6,
        label=label,
        edgecolor="white",
    )
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reliability diagram to {out_file}")

    plt.close(fig)

    result = {
        "calibrated": curve_cal,
        "save_path": str(save_path) if save_path else None,
    }
    if curve_uncal is not None:
        result["uncalibrated"] = curve_uncal
    return result


class DirectionalCalibrator:
    """
    Directional probability calibrator.
    Calibrates out-of-fold (OOF) directional probability forecasts using:
    - Isotonic Regression if N >= 1000 samples (non-parametric, preserves monotonic order)
    - Platt Scaling (Logistic Regression on logit) if N < 1000 samples (parametric, prevents overfitting)

    Strictly no future look-ahead: fit must be called only on past/OOF data.
    """

    def __init__(
        self,
        method: str = "auto",
        n_bins: int = 10,
        min_prob: float = 1e-4,
        max_prob: float = 1.0 - 1e-4,
    ):
        """
        Parameters
        ----------
        method : {'auto', 'isotonic', 'platt'}, default='auto'
            Calibration method.
        n_bins : int, default=10
            Number of bins for ECE reporting.
        min_prob : float, default=1e-4
            Lower probability clamp.
        max_prob : float, default=1 - 1e-4
            Upper probability clamp.
        """
        if method not in ("auto", "isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.n_bins = n_bins
        self.min_prob = min_prob
        self.max_prob = max_prob

        self.method_used_: Optional[str] = None
        self.model_: Any = None
        self.is_fitted_: bool = False
        self.n_samples_: int = 0
        self.metrics_oof_: Dict[str, Any] = {}

    def fit(
        self,
        y_prob_oof: np.ndarray,
        y_true: np.ndarray,
    ) -> "DirectionalCalibrator":
        """
        Fit the calibrator on out-of-fold predictions.

        Parameters
        ----------
        y_prob_oof : np.ndarray
            Out-of-fold predicted probabilities in [0, 1].
        y_true : np.ndarray
            Binary true direction indicator in {0, 1}.
        """
        y_p = np.asarray(y_prob_oof, dtype=float).ravel()
        y_t = np.asarray(y_true, dtype=float).ravel()

        if len(y_p) != len(y_t):
            raise ValueError(f"Mismatched shapes: y_prob_oof={len(y_p)}, y_true={len(y_t)}")
        if len(y_p) < 10:
            raise ValueError(f"Insufficient calibration samples: {len(y_p)} (minimum 10 required)")

        # Verify binary
        unique_targets = set(np.unique(y_t))
        if not unique_targets.issubset({0.0, 1.0}):
            raise ValueError(f"y_true must be binary {0, 1}, got {unique_targets}")

        n = len(y_p)
        self.n_samples_ = n

        # Choose method
        if self.method == "auto":
            self.method_used_ = "isotonic" if n >= 1000 else "platt"
        else:
            self.method_used_ = self.method

        # Clip input probs slightly away from 0/1 for numerical stability
        y_p_clipped = np.clip(y_p, self.min_prob, self.max_prob)

        if self.method_used_ == "isotonic":
            iso = IsotonicRegression(
                out_of_bounds="clip",
                y_min=self.min_prob,
                y_max=self.max_prob,
                increasing=True,
            )
            iso.fit(y_p_clipped, y_t)
            self.model_ = iso
        else:
            # Platt scaling: LogisticRegression on logit
            # logit(p) = ln(p / (1 - p))
            logits = logit(y_p_clipped).reshape(-1, 1)
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )
            clf.fit(logits, y_t)
            self.model_ = clf

        self.is_fitted_ = True

        # Calculate and store OOF evaluation
        calibrated_oof = self.predict_proba(y_p)
        self.metrics_oof_ = self.evaluate(y_true=y_t, y_prob_uncal=y_p, y_prob_cal=calibrated_oof)
        logger.info(
            f"Fitted DirectionalCalibrator using {self.method_used_} on {n} samples. "
            f"Brier: {self.metrics_oof_['uncalibrated']['brier_score']:.4f} -> "
            f"{self.metrics_oof_['calibrated']['brier_score']:.4f}, "
            f"ECE: {self.metrics_oof_['uncalibrated']['ece']:.4f} -> "
            f"{self.metrics_oof_['calibrated']['ece']:.4f}"
        )
        return self

    def predict_proba(self, y_prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calibrate uncalibrated probabilities.

        Parameters
        ----------
        y_prob : float or np.ndarray
            Uncalibrated probability or array of probabilities.

        Returns
        -------
        float or np.ndarray
            Calibrated probability matching input shape.
        """
        if not self.is_fitted_:
            raise RuntimeError("DirectionalCalibrator is not fitted yet. Call fit() first.")

        is_scalar = np.isscalar(y_prob)
        p = np.asarray(y_prob, dtype=float)
        p_clipped = np.clip(p, self.min_prob, self.max_prob)

        if self.method_used_ == "isotonic":
            calibrated = self.model_.predict(p_clipped.ravel()).reshape(p.shape)
        else:
            logits = logit(p_clipped.ravel()).reshape(-1, 1)
            # Probability of class 1
            calibrated = self.model_.predict_proba(logits)[:, 1].reshape(p.shape)

        calibrated = np.clip(calibrated, self.min_prob, self.max_prob)
        if is_scalar:
            return float(calibrated.item())
        return calibrated

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob_uncal: np.ndarray,
        y_prob_cal: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate metrics before and after calibration.
        """
        if y_prob_cal is None:
            if not self.is_fitted_:
                raise RuntimeError("Must provide y_prob_cal or fit the calibrator first.")
            y_prob_cal = self.predict_proba(y_prob_uncal)

        m_uncal = compute_reliability_curve(y_true, y_prob_uncal, n_bins=self.n_bins)
        m_cal = compute_reliability_curve(y_true, y_prob_cal, n_bins=self.n_bins)

        brier_diff = m_cal["brier_score"] - m_uncal["brier_score"]
        ece_diff = m_cal["ece"] - m_uncal["ece"]
        loss_diff = m_cal["log_loss"] - m_uncal["log_loss"]

        return {
            "method": self.method_used_,
            "n_samples": len(y_true),
            "uncalibrated": m_uncal,
            "calibrated": m_cal,
            "improvement": {
                "brier_score_delta": float(brier_diff),
                "ece_delta": float(ece_diff),
                "log_loss_delta": float(loss_diff),
                "brier_improved": bool(brier_diff <= 0),
                "ece_improved": bool(ece_diff <= 0),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator state to dictionary."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot serialize unfitted DirectionalCalibrator")
        state: Dict[str, Any] = {
            "method": self.method,
            "method_used": self.method_used_,
            "n_samples": self.n_samples_,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
            "metrics_oof": {
                k: v for k, v in self.metrics_oof_.items() if k != "uncalibrated" and k != "calibrated"
            },
        }
        if self.method_used_ == "platt":
            state["coef"] = self.model_.coef_.tolist()
            state["intercept"] = self.model_.intercept_.tolist()
            state["classes"] = self.model_.classes_.tolist()
        elif self.method_used_ == "isotonic":
            state["x_thresholds"] = self.model_.X_thresholds_.tolist()
            state["y_thresholds"] = self.model_.y_thresholds_.tolist()
        return state

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectionalCalibrator":
        """Deserialize calibrator from dictionary."""
        calibrator = cls(
            method=data["method"],
            min_prob=data["min_prob"],
            max_prob=data["max_prob"],
        )
        calibrator.method_used_ = data["method_used"]
        calibrator.n_samples_ = data["n_samples"]
        calibrator.metrics_oof_ = data.get("metrics_oof", {})

        if calibrator.method_used_ == "platt":
            clf = LogisticRegression()
            clf.coef_ = np.array(data["coef"])
            clf.intercept_ = np.array(data["intercept"])
            clf.classes_ = np.array(data["classes"])
            calibrator.model_ = clf
        elif calibrator.method_used_ == "isotonic":
            iso = IsotonicRegression(
                out_of_bounds="clip",
                y_min=calibrator.min_prob,
                y_max=calibrator.max_prob,
                increasing=True,
            )
            iso.X_thresholds_ = np.array(data["x_thresholds"])
            iso.y_thresholds_ = np.array(data["y_thresholds"])
            iso.f_ = lambda x: np.interp(x, iso.X_thresholds_, iso.y_thresholds_)
            calibrator.model_ = iso

        calibrator.is_fitted_ = True
        return calibrator
