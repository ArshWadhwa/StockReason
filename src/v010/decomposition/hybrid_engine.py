"""
StockReason v0.10 — Hybrid Return Decomposition Engine (Market + Sector + Stock).
=================================================================================
Remediates Phase 2 (Section 7):
Decomposes total future stock return into three orthogonal components:
    r_{i, t, H} = beta_{i, m} * r_{m, H} + beta_{i, s} * r_{s, H}^{resid} + r_{i, H}^{idio}

1. Component A (Market): Predicts expected future broad market return (Nifty 50).
   - Inputs: trailing returns, realized vol, India VIX, market breadth, market slope.
   - Evaluates: No-change (0.0), Historical drift, Ridge, LightGBM.
2. Component B (Sector): Predicts expected sector excess return over market.
   - Evaluates Banking, IT, Pharma, Auto, FMCG relative performance.
3. Component C (Stock Idiosyncratic): Predicts residual stock performance.
   - Fits Ridge and LightGBM directly on orthogonalized residuals.
4. Tri-Part Recomposition:
   - Evaluates absolute return and INR price error:
     P_hat_{i, t+H} = P_{i, t} * (1.0 + r_hat_{i, t, H})
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb

from src.v091.features.pipeline import FEATURE_COLS_21
from src.v092.models.baselines import compute_absolute_price_metrics


class HybridReturnDecompositionEngine:
    def __init__(
        self,
        market_model_type: str = "lgbm",
        idiosyncratic_model_type: str = "lgbm",
        horizon: int = 21,
        seed: int = 42,
    ):
        self.market_model_type = market_model_type
        self.idiosyncratic_model_type = idiosyncratic_model_type
        self.horizon = horizon
        self.seed = seed

        self.market_model = None
        self.sector_models: Dict[str, Any] = {}
        self.idio_model = None
        self.market_drift: float = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        features: List[str] = FEATURE_COLS_21,
    ):
        """
        Fits Component A (Market), Component B (Sector), and Component C (Stock Idiosyncratic).
        """
        target_col = f"target_return_{self.horizon}d"
        mkt_col = "nifty_ret_21d" if "nifty_ret_21d" in train_df.columns else "nifty_ret_1d"

        clean_tr = train_df.dropna(subset=features + [target_col, "close"]).copy()
        
        # ---------------------------------------------------------------------
        # 1. Component A: Market Index Forecasting
        # ---------------------------------------------------------------------
        # Group by date to get daily unique market return series
        daily_mkt = clean_tr.groupby("date_str").first().reset_index()
        mkt_features = [c for c in ["nifty_ret_21d", "nifty_ret_1d", "nifty_trend_slope", "vix_level"] if c in daily_mkt.columns]
        
        if mkt_features and mkt_col in daily_mkt.columns:
            X_mkt = daily_mkt[mkt_features].values
            y_mkt = daily_mkt[mkt_col].values
            self.market_drift = float(np.mean(y_mkt))

            if self.market_model_type == "lgbm":
                self.market_model = lgb.LGBMRegressor(
                    learning_rate=0.03, n_estimators=50, num_leaves=15, max_depth=3,
                    random_state=self.seed, n_jobs=1, verbose=-1
                ).fit(X_mkt, y_mkt)
            else:
                self.market_model = Ridge(alpha=100.0, random_state=self.seed).fit(X_mkt, y_mkt)
        else:
            self.market_drift = 0.015

        # ---------------------------------------------------------------------
        # 2. Estimate Betas & Residuals on Training Data
        # ---------------------------------------------------------------------
        # beta_market: use stock beta feature if available, else trailing cov
        if "beta_60d" in clean_tr.columns:
            beta_m = clean_tr["beta_60d"].clip(0.1, 2.5).values
        else:
            beta_m = np.ones(len(clean_tr))

        mkt_train_ret = clean_tr[mkt_col].values if mkt_col in clean_tr.columns else np.zeros(len(clean_tr))
        
        # Expected market contribution
        mkt_contrib = beta_m * mkt_train_ret

        # Sector residual contribution if available
        sector_contrib = np.zeros(len(clean_tr))
        if "sector" in clean_tr.columns:
            sec_means = clean_tr.groupby("sector")[target_col].transform("mean").values
            sector_contrib = sec_means - mkt_train_ret

        # Idiosyncratic residual target: r_idio = r - beta_m*r_m - r_s
        clean_tr["target_idio"] = clean_tr[target_col].values - mkt_contrib - sector_contrib

        # ---------------------------------------------------------------------
        # 3. Component C: Idiosyncratic Residual Modeling
        # ---------------------------------------------------------------------
        X_tr = clean_tr[features].values
        y_tr_idio = clean_tr["target_idio"].values

        if self.idiosyncratic_model_type == "lgbm":
            self.idio_model = lgb.LGBMRegressor(
                learning_rate=0.03, n_estimators=100, num_leaves=31, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=self.seed,
                n_jobs=1, verbose=-1
            ).fit(X_tr, y_tr_idio)
        elif self.idiosyncratic_model_type == "elastic_net":
            self.idio_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.seed).fit(X_tr, y_tr_idio)
        else:
            self.idio_model = Ridge(alpha=100.0, random_state=self.seed).fit(X_tr, y_tr_idio)

    def predict(
        self,
        eval_df: pd.DataFrame,
        features: List[str] = FEATURE_COLS_21,
    ) -> Dict[str, np.ndarray]:
        """
        Recomposes total future return:
        r_hat = beta_m * r_hat_m + r_hat_s + r_hat_idio
        P_hat = P_t * (1 + r_hat)
        """
        target_col = f"target_return_{self.horizon}d"
        mkt_col = "nifty_ret_21d" if "nifty_ret_21d" in eval_df.columns else "nifty_ret_1d"
        mkt_features = [c for c in ["nifty_ret_21d", "nifty_ret_1d", "nifty_trend_slope", "vix_level"] if c in eval_df.columns]

        # 1. Market forecast
        if self.market_model is not None and mkt_features:
            daily_mkt = eval_df.groupby("date_str")[mkt_features].first()
            pred_mkt_daily = pd.Series(self.market_model.predict(daily_mkt.values), index=daily_mkt.index)
            pred_mkt = eval_df["date_str"].map(pred_mkt_daily).fillna(self.market_drift).values
        else:
            pred_mkt = np.full(len(eval_df), self.market_drift)

        # 2. Beta exposures
        if "beta_60d" in eval_df.columns:
            beta_m = eval_df["beta_60d"].clip(0.1, 2.5).values
        else:
            beta_m = np.ones(len(eval_df))

        # 3. Sector residual forecast (simple shrinkage)
        sector_resid = np.zeros(len(eval_df))
        if "sector" in eval_df.columns and "sector_rel_ret_21d" in eval_df.columns:
            sector_resid = (eval_df["sector_rel_ret_21d"].fillna(0.0) * 0.1).values

        # 4. Idiosyncratic forecast
        X_val = eval_df[features].values
        if self.idio_model is not None:
            pred_idio = self.idio_model.predict(X_val)
        else:
            pred_idio = np.zeros(len(eval_df))

        # 5. Hybrid recomposition
        pred_total_return = (beta_m * pred_mkt) + sector_resid + pred_idio
        pred_inr_price = eval_df["close"].values * (1.0 + pred_total_return)

        return {
            "pred_total_return": pred_total_return,
            "pred_inr_price": pred_inr_price,
            "component_market": beta_m * pred_mkt,
            "component_sector": sector_resid,
            "component_idiosyncratic": pred_idio,
        }

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        features: List[str] = FEATURE_COLS_21,
    ) -> Dict[str, Any]:
        """
        Evaluates the hybrid decomposition model on Return MAE, RMSE, INR Price MAE, MAPE %.
        """
        target_col = f"target_return_{self.horizon}d"
        clean_eval = eval_df.dropna(subset=features + [target_col, "close"]).copy()

        preds = self.predict(clean_eval, features=features)
        y_true = clean_eval[target_col].values
        y_pred = preds["pred_total_return"]
        prices = clean_eval["close"].values

        metrics = compute_absolute_price_metrics(y_true, y_pred, prices)
        return {
            "model_name": f"Hybrid Decomposition ({self.market_model_type.upper()} Market + {self.idiosyncratic_model_type.upper()} Idio)",
            "horizon": f"{self.horizon}d",
            "eval_observations": len(clean_eval),
            **metrics,
        }
