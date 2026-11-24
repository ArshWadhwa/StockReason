"""Conformal Prediction intervals and uncertainty calibration for StockReason v0.7.

Implements:
1. Split-Conformal Prediction (homogeneous & locally normalized residuals).
2. Conformalized Quantile Regression (CQR) for adaptive, state-dependent interval widths.
3. Adaptive Conformal Inference (ACI) & Rolling-Window calibration for non-stationary market regimes.
4. Winkler Interval Score and empirical coverage diagnostics (per-regime and over time).
5. Benchmark comparison against legacy MC-dropout intervals (mean ± 1.96·std).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


def compute_conformal_quantile(
    scores: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Compute the finite-sample conformal quantile:
    q = Quantile(scores, ceil((n + 1) * (1 - alpha)) / n)

    Parameters
    ----------
    scores : np.ndarray
        Calibration non-conformity scores (e.g. absolute residuals).
    alpha : float, default=0.20
        Miscoverage rate (nominal coverage = 1 - alpha = 0.80 for 80%).

    Returns
    -------
    float
        Conformal quantile value.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = len(s)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores array.")

    # Conformal finite-sample correction level
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    q = float(np.quantile(s, level, method="higher"))
    return q


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Compute empirical coverage fraction: (1 / N) * sum(lower <= y_true <= upper).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    l = np.asarray(lower, dtype=float).ravel()
    u = np.asarray(upper, dtype=float).ravel()
    if not (len(y) == len(l) == len(u)):
        raise ValueError(f"Lengths mismatch: y={len(y)}, lower={len(l)}, upper={len(u)}")
    if len(y) == 0:
        return 0.0
    covered = (y >= l) & (y <= u)
    return float(np.mean(covered))


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Compute Winkler interval score (proper scoring rule for prediction intervals).
    Lower is better.

    Score = (upper - lower) + (2 / alpha) * (lower - y) * I(y < lower)
                            + (2 / alpha) * (y - upper) * I(y > upper)
    """
    y = np.asarray(y_true, dtype=float).ravel()
    l = np.asarray(lower, dtype=float).ravel()
    u = np.asarray(upper, dtype=float).ravel()
    if not (len(y) == len(l) == len(u)):
        raise ValueError("Lengths mismatch")
    if len(y) == 0:
        return 0.0

    width = u - l
    penalty_low = np.maximum(0.0, l - y) * (2.0 / alpha)
    penalty_high = np.maximum(0.0, y - u) * (2.0 / alpha)
    scores = width + penalty_low + penalty_high
    return float(np.mean(scores))


def compute_interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute comprehensive prediction interval metrics.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    l = np.asarray(lower, dtype=float).ravel()
    u = np.asarray(upper, dtype=float).ravel()
    widths = u - l

    coverage = compute_coverage(y, l, u)
    nominal = 1.0 - alpha
    gap = coverage - nominal

    return {
        "n_samples": len(y),
        "nominal_coverage": float(nominal),
        "empirical_coverage": float(coverage),
        "coverage_gap": float(gap),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "std_width": float(np.std(widths)),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "winkler_score": compute_winkler_score(y, l, u, alpha=alpha),
    }


def evaluate_coverage_by_regime(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    regimes: np.ndarray,
    alpha: float = 0.20,
) -> Dict[str, Dict[str, Any]]:
    """
    Break down coverage and interval width by market regime.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    l = np.asarray(lower, dtype=float).ravel()
    u = np.asarray(upper, dtype=float).ravel()
    regs = np.asarray(regimes).ravel()

    results: Dict[str, Dict[str, Any]] = {}
    for r in np.unique(regs):
        idx = (regs == r)
        if np.sum(idx) == 0:
            continue
        sub_metrics = compute_interval_metrics(y[idx], l[idx], u[idx], alpha=alpha)
        results[str(r)] = sub_metrics
    return results


def evaluate_coverage_over_time(
    dates: np.ndarray,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rolling_window: int = 60,
    alpha: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute rolling-window empirical coverage over time.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    l = np.asarray(lower, dtype=float).ravel()
    u = np.asarray(upper, dtype=float).ravel()
    d = np.asarray(dates).ravel()

    covered = ((y >= l) & (y <= u)).astype(float)
    n = len(covered)
    if n < rolling_window:
        return {
            "dates": [str(x)[:10] for x in d],
            "rolling_coverage": [float(np.mean(covered))] * n,
            "nominal_coverage": 1.0 - alpha,
        }

    rolling_cov: List[float] = []
    out_dates: List[str] = []
    for i in range(rolling_window, n + 1):
        win_cov = float(np.mean(covered[i - rolling_window : i]))
        rolling_cov.append(win_cov)
        out_dates.append(str(d[i - 1])[:10])

    return {
        "dates": out_dates,
        "rolling_coverage": rolling_cov,
        "nominal_coverage": 1.0 - alpha,
        "window_size": rolling_window,
    }


class SplitConformalRegressor:
    """
    Standard Split-Conformal Prediction Regressor.

    Calibrates on out-of-fold / holdout residuals |y - y_hat|.
    Guarantees finite-sample marginal coverage >= 1 - alpha under exchangeability.
    """

    def __init__(self, alpha: float = 0.20, normalized: bool = False):
        """
        Parameters
        ----------
        alpha : float, default=0.20
            Miscoverage rate (nominal coverage = 80%).
        normalized : bool, default=False
            If True, uses normalized scores |y - y_hat| / sigma.
        """
        self.alpha = alpha
        self.nominal_coverage = 1.0 - alpha
        self.normalized = normalized
        self.q_hat_: Optional[float] = None
        self.is_fitted_: bool = False
        self.n_calibration_samples_: int = 0

    def fit(
        self,
        y_true_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        sigma_cal: Optional[np.ndarray] = None,
    ) -> "SplitConformalRegressor":
        """
        Calibrate on holdout residuals.
        """
        y = np.asarray(y_true_cal, dtype=float).ravel()
        y_hat = np.asarray(y_pred_cal, dtype=float).ravel()
        if len(y) != len(y_hat):
            raise ValueError("Mismatched calibration lengths")
        if len(y) == 0:
            raise ValueError("Empty calibration set")

        residuals = np.abs(y - y_hat)
        if self.normalized:
            if sigma_cal is None:
                raise ValueError("sigma_cal required when normalized=True")
            sigmas = np.asarray(sigma_cal, dtype=float).ravel()
            sigmas = np.where(sigmas <= 1e-8, 1e-4, sigmas)
            scores = residuals / sigmas
        else:
            scores = residuals

        self.q_hat_ = compute_conformal_quantile(scores, alpha=self.alpha)
        self.n_calibration_samples_ = len(scores)
        self.is_fitted_ = True
        logger.info(
            f"Fitted SplitConformalRegressor(alpha={self.alpha}, normalized={self.normalized}): "
            f"n={self.n_calibration_samples_}, q_hat={self.q_hat_:.6f}"
        )
        return self

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray],
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute conformal prediction interval [lower, upper].
        """
        if not self.is_fitted_ or self.q_hat_ is None:
            raise RuntimeError("SplitConformalRegressor is not fitted yet.")

        is_scalar = np.isscalar(y_pred)
        pred = np.asarray(y_pred, dtype=float)

        if self.normalized:
            if sigma is None:
                raise ValueError("sigma is required when normalized=True")
            s = np.asarray(sigma, dtype=float)
            margin = self.q_hat_ * s
        else:
            margin = self.q_hat_

        lower = pred - margin
        upper = pred + margin

        if is_scalar:
            return float(lower.item()), float(upper.item())
        return lower, upper


class ConformalizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR) (Romano, Patterson, Candès 2019).

    Fits or wraps base quantile regressors for lower quantile (alpha/2)
    and upper quantile (1 - alpha/2), then computes the conformal adjustment
    constant Q from calibration non-conformity scores:
      E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
    Calibrated interval: [q_low(x) - Q, q_high(x) + Q].
    """

    def __init__(
        self,
        alpha: float = 0.20,
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        alpha : float, default=0.20
            Nominal miscoverage rate (nominal coverage 1 - alpha = 80%).
            Quantiles used: alpha/2 = 0.10, 1 - alpha/2 = 0.90.
        """
        self.alpha = alpha
        self.nominal_coverage = 1.0 - alpha
        self.q_low_alpha = alpha / 2.0
        self.q_high_alpha = 1.0 - (alpha / 2.0)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_low_: Optional[GradientBoostingRegressor] = None
        self.model_high_: Optional[GradientBoostingRegressor] = None
        self.q_conformal_: Optional[float] = None
        self.is_fitted_: bool = False
        self.n_calibration_samples_: int = 0

    def fit_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> "ConformalizedQuantileRegression":
        """
        Fit base quantile gradient boosting regressors on training split.
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).ravel()

        self.model_low_ = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.q_low_alpha,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_high_ = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.q_high_alpha,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self.model_low_.fit(X, y)
        self.model_high_.fit(X, y)
        return self

    def calibrate(
        self,
        y_true_cal: np.ndarray,
        q_low_cal: Optional[np.ndarray] = None,
        q_high_cal: Optional[np.ndarray] = None,
        X_cal: Optional[np.ndarray] = None,
    ) -> "ConformalizedQuantileRegression":
        """
        Calibrate CQR adjustment on holdout / calibration set.
        Can provide either (X_cal) if base models were fitted,
        or directly provide pre-computed (q_low_cal, q_high_cal).
        """
        y = np.asarray(y_true_cal, dtype=float).ravel()

        if q_low_cal is None or q_high_cal is None:
            if X_cal is None:
                raise ValueError("Must provide either (q_low_cal, q_high_cal) or X_cal.")
            if self.model_low_ is None or self.model_high_ is None:
                raise RuntimeError("Base models not fitted yet; call fit_base_models(X, y) first.")
            q_low = self.model_low_.predict(X_cal)
            q_high = self.model_high_.predict(X_cal)
        else:
            q_low = np.asarray(q_low_cal, dtype=float).ravel()
            q_high = np.asarray(q_high_cal, dtype=float).ravel()

        if not (len(y) == len(q_low) == len(q_high)):
            raise ValueError("Mismatched calibration lengths")

        # CQR conformity score: E_i = max(q_low - y, y - q_high)
        scores = np.maximum(q_low - y, y - q_high)
        self.q_conformal_ = compute_conformal_quantile(scores, alpha=self.alpha)
        self.n_calibration_samples_ = len(scores)
        self.is_fitted_ = True
        logger.info(
            f"Fitted CQR: n={self.n_calibration_samples_}, adjustment Q={self.q_conformal_:.6f}"
        )
        return self

    def predict_interval(
        self,
        X_test: Optional[np.ndarray] = None,
        q_low: Optional[Union[float, np.ndarray]] = None,
        q_high: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute CQR prediction intervals:
        lower = q_low - Q
        upper = q_high + Q
        """
        if not self.is_fitted_ or self.q_conformal_ is None:
            raise RuntimeError("CQR is not calibrated yet.")

        if q_low is None or q_high is None:
            if X_test is None:
                raise ValueError("Must provide either (q_low, q_high) or X_test.")
            if self.model_low_ is None or self.model_high_ is None:
                raise RuntimeError("Base models are missing.")
            ql = self.model_low_.predict(X_test)
            qh = self.model_high_.predict(X_test)
        else:
            ql = q_low
            qh = q_high

        is_scalar = np.isscalar(ql)
        lower = np.asarray(ql, dtype=float) - self.q_conformal_
        upper = np.asarray(qh, dtype=float) + self.q_conformal_

        # Enforce ordering lower <= upper
        lower_adj = np.minimum(lower, upper)
        upper_adj = np.maximum(lower, upper)

        if is_scalar:
            return float(lower_adj.item()), float(upper_adj.item())
        return lower_adj, upper_adj


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) (Gibbs & Candès 2021).

    Maintains long-run empirical coverage under non-stationary regimes
    by dynamically updating nominal miscoverage alpha_t:
      alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1 if y_t not in C_t else 0.
    """

    def __init__(
        self,
        target_alpha: float = 0.20,
        gamma: float = 0.05,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        window_size: int = 126,
    ):
        """
        Parameters
        ----------
        target_alpha : float, default=0.20
            Target nominal miscoverage rate (80% coverage).
        gamma : float, default=0.05
            Learning rate step-size for ACI updates.
        alpha_min : float, default=0.01
            Minimum allowable alpha_t.
        alpha_max : float, default=0.99
            Maximum allowable alpha_t.
        window_size : int, default=126
            Number of recent past residuals used to evaluate conformal quantile.
        """
        self.target_alpha = target_alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.window_size = window_size

        self.current_alpha_: float = target_alpha
        self.residual_history_: List[float] = []
        self.err_history_: List[int] = []
        self.alpha_history_: List[float] = []

    def reset(self, initial_residuals: Optional[np.ndarray] = None) -> None:
        """Reset ACI state with optional initial warm-up calibration residuals."""
        self.current_alpha_ = self.target_alpha
        self.err_history_ = []
        self.alpha_history_ = [self.current_alpha_]
        if initial_residuals is not None:
            self.residual_history_ = [float(r) for r in initial_residuals]
        else:
            self.residual_history_ = []

    def predict_interval(self, y_pred: float) -> Tuple[float, float]:
        """
        Produce prediction interval for next time-step given current alpha_t.
        Uses rolling window of past residuals.
        """
        if len(self.residual_history_) < 5:
            # Cold-start fallback: heuristic default margin
            margin = 0.03
        else:
            recent_scores = np.array(self.residual_history_[-self.window_size :])
            margin = compute_conformal_quantile(recent_scores, alpha=self.current_alpha_)

        lower = y_pred - margin
        upper = y_pred + margin
        return float(lower), float(upper)

    def step(self, y_true: float, lower: float, upper: float, y_pred: float) -> Dict[str, Any]:
        """
        Observe true value y_t, record error, update residual history and alpha_{t+1}.
        Strictly causal: updates happen AFTER interval was evaluated.
        """
        # 1. Evaluate coverage
        is_covered = (y_true >= lower) and (y_true <= upper)
        err = 0 if is_covered else 1
        self.err_history_.append(err)

        # 2. Add observed residual
        res = abs(y_true - y_pred)
        self.residual_history_.append(float(res))

        # 3. Update alpha_{t+1}
        # alpha_{t+1} = alpha_t + gamma * (alpha_target - err)
        # If miscovered (err=1), alpha decreases -> larger interval
        # If covered (err=0), alpha increases -> tighter interval
        next_alpha = self.current_alpha_ + self.gamma * (self.target_alpha - err)
        self.current_alpha_ = float(np.clip(next_alpha, self.alpha_min, self.alpha_max))
        self.alpha_history_.append(self.current_alpha_)

        return {
            "covered": is_covered,
            "err": err,
            "current_alpha": self.current_alpha_,
            "observed_residual": float(res),
        }


def run_rolling_conformal_simulation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 126,
    alpha: float = 0.20,
    adaptive_aci: bool = True,
    gamma: float = 0.05,
) -> Dict[str, Any]:
    """
    Simulate rolling-window conformal calibration over a time series.
    Strictly causal / no lookahead:
    At step t, interval is computed using only data from {t-W, ..., t-1}.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values.
    y_pred : np.ndarray
        Point predictions.
    window_size : int, default=126
        Calibration window length.
    alpha : float, default=0.20
        Nominal miscoverage rate (80% coverage).
    adaptive_aci : bool, default=True
        Whether to use ACI dynamic alpha adaptation.
    gamma : float, default=0.05
        ACI learning rate.

    Returns
    -------
    Dict[str, Any]
        Dictionary with simulation metrics, lowers, uppers, and coverage.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_t)
    if n <= window_size:
        raise ValueError(f"Series length {n} must be greater than window_size {window_size}")

    lowers: List[float] = []
    uppers: List[float] = []
    eval_y: List[float] = []
    alphas: List[float] = []

    aci = AdaptiveConformalInference(target_alpha=alpha, gamma=gamma, window_size=window_size)
    init_res = np.abs(y_t[:window_size] - y_p[:window_size])
    aci.reset(initial_residuals=init_res)

    for t in range(window_size, n):
        pred_val = float(y_p[t])
        true_val = float(y_t[t])

        if adaptive_aci:
            l, u = aci.predict_interval(pred_val)
            aci.step(y_true=true_val, lower=l, upper=u, y_pred=pred_val)
            alphas.append(aci.current_alpha_)
        else:
            # Fixed alpha rolling split conformal
            past_scores = np.abs(y_t[t - window_size : t] - y_p[t - window_size : t])
            q = compute_conformal_quantile(past_scores, alpha=alpha)
            l = pred_val - q
            u = pred_val + q
            alphas.append(alpha)

        lowers.append(l)
        uppers.append(u)
        eval_y.append(true_val)

    low_arr = np.array(lowers)
    up_arr = np.array(uppers)
    y_eval_arr = np.array(eval_y)

    metrics = compute_interval_metrics(y_eval_arr, low_arr, up_arr, alpha=alpha)
    return {
        "metrics": metrics,
        "lowers": low_arr,
        "uppers": up_arr,
        "y_eval": y_eval_arr,
        "alphas": alphas,
    }


def compare_intervals_with_mc_dropout(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    mc_std: np.ndarray,
    lower_conformal: np.ndarray,
    upper_conformal: np.ndarray,
    nominal_conformal: float = 0.80,
    nominal_mc: float = 0.95,
) -> Dict[str, Any]:
    """
    Compare empirical performance of conformal prediction intervals
    against legacy MC-dropout intervals (mean ± 1.96·std).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth realized targets.
    point_pred : np.ndarray
        Model point predictions (mean).
    mc_std : np.ndarray
        Monte Carlo Dropout standard deviation.
    lower_conformal : np.ndarray
        Calibrated conformal lower bound.
    upper_conformal : np.ndarray
        Calibrated conformal upper bound.
    nominal_conformal : float, default=0.80
        Nominal coverage of conformal interval.
    nominal_mc : float, default=0.95
        Nominal coverage claimed by 1.96 * std (95%).

    Returns
    -------
    Dict[str, Any]
        Side-by-side comparison dictionary.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(point_pred, dtype=float).ravel()
    s = np.asarray(mc_std, dtype=float).ravel()

    # Legacy MC-dropout nominal 95% interval
    lower_mc = p - 1.96 * s
    upper_mc = p + 1.96 * s

    mc_metrics = compute_interval_metrics(y, lower_mc, upper_mc, alpha=1.0 - nominal_mc)
    conf_metrics = compute_interval_metrics(
        y, lower_conformal, upper_conformal, alpha=1.0 - nominal_conformal
    )

    finding = (
        f"MC-dropout nominally claimed {nominal_mc*100:.0f}% coverage but achieved "
        f"{mc_metrics['empirical_coverage']*100:.1f}% (gap: {mc_metrics['coverage_gap']*100:+.1f}%). "
        f"Conformal achieved {conf_metrics['empirical_coverage']*100:.1f}% for nominal "
        f"{nominal_conformal*100:.0f}% coverage (gap: {conf_metrics['coverage_gap']*100:+.1f}%)."
    )

    return {
        "finding": finding,
        "mc_dropout": mc_metrics,
        "conformal": conf_metrics,
        "mc_undercoverage_deficit": float(nominal_mc - mc_metrics["empirical_coverage"]),
        "conformal_calibration_gap": float(abs(conf_metrics["coverage_gap"])),
    }
