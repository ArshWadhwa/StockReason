"""StockReason — Production API Service & Transparent Intelligence Engine."""

from typing import Dict, Any, Optional
import logging
import numpy as np

from src.inference.predictor import StockReasonPredictor
from api.data_hub import StockReasonDataHub, _normalize_ticker

logger = logging.getLogger("stockreason_service")


class StockReasonAPIService:
    """
    Transparent API Service enforcing fail-safe fallback,
    nominal-price reconstruction, and exchange calendar compliance.
    """

    def __init__(
        self,
        baseline_price_mae: float = 103.40,
        model_version: str = "0.14.0",
        is_probability_calibrated: bool = False,
        data_hub: Optional[StockReasonDataHub] = None,
    ):
        self.baseline_price_mae = baseline_price_mae
        self.model_version = model_version
        self.is_probability_calibrated = is_probability_calibrated
        self.data_hub = data_hub or StockReasonDataHub()
        self.predictor = StockReasonPredictor(
            model_type="baseline",
            is_probability_calibrated=is_probability_calibrated,
        )
        self._predictors: Dict[str, StockReasonPredictor] = {"baseline": self.predictor}

    def _get_predictor(self, model_type: str) -> tuple[StockReasonPredictor, Optional[str]]:
        mt = (model_type or "baseline").lower()
        if mt in self._predictors:
            return self._predictors[mt], None
        try:
            pred = StockReasonPredictor(
                model_type=mt,
                is_probability_calibrated=self.is_probability_calibrated,
            )
            self._predictors[mt] = pred
            return pred, None
        except Exception as exc:
            logger.warning("Could not initialize model_type=%s (%s); falling back to baseline.", mt, exc)
            return self.predictor, f"Model '{mt}' unavailable in runtime ({exc}); fell back to NoChangeRandomWalk."

    def generate_forecast(
        self,
        symbol: str,
        current_price: float,
        features: Optional[Dict[str, float]] = None,
        reference_date: Optional[str] = None,
        horizon_days: int = 21,
        model_type: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Generates production forecast following transparency schema.
        Supports baseline, lightgbm, and catboost with safe fallback.
        """
        mt = (model_type or "baseline").lower()
        predictor, init_warning = self._get_predictor(mt)

        # Auto-hydrate features from local feature store when user selects ML model without explicit features
        effective_features = features
        hydrated_used = False
        if effective_features is None and mt in ("lightgbm", "catboost", "best"):
            model_obj = predictor.lgb_model if mt in ("lightgbm", "best") else predictor.cb_model
            if model_obj is not None and getattr(model_obj, "feature_names", None):
                effective_features = self.data_hub.get_latest_features_for_model(
                    symbol, model_obj.feature_names
                )
                hydrated_used = effective_features is not None

        # CatBoost direct inference support if requested & available
        if mt == "catboost" and predictor.cb_model is not None and effective_features is not None:
            resp = predictor.predict_single(
                symbol=symbol,
                close=current_price,
                features=None,
                reference_date=reference_date,
                horizon=horizon_days,
            )
            feats = predictor.cb_model.feature_names
            if feats and all(f in effective_features for f in feats):
                X = np.array([[effective_features[f] for f in feats]], dtype=np.float32)
                pred_ret = float(predictor.cb_model.predict(X)[0])
                resp.predicted_return = round(pred_ret, 4)
                resp.predicted_price = round(current_price * (1.0 + pred_ret), 2)
                resp.model_name = "CatBoost_21d"
                resp.model_status = "EXPERIMENTAL_NOT_PROMOTED"
                resp.warnings.insert(
                    0,
                    "Model is experimental; CatBoost residual return does not beat No-Change baseline on INR MAE.",
                )
        else:
            resp = predictor.predict_single(
                symbol=symbol,
                close=current_price,
                features=effective_features,
                reference_date=reference_date,
                horizon=horizon_days,
            )

        data = resp.model_dump()
        if init_warning:
            data["warnings"].insert(0, init_warning)
        if hydrated_used:
            data["warnings"].append(
                f"Auto-hydrated latest historical features for {_normalize_ticker(symbol)} from local store."
            )

        # Attach conformal intervals if ticker is available in local history
        try:
            risk = self.data_hub.get_stock_risk_profile(symbol)
            scale = float(np.sqrt(max(1, horizon_days) / 21.0))
            for item in risk.get("conformal_intervals", []):
                low_p = round(current_price * (1.0 + item["return_lower"] * scale), 2)
                high_p = round(current_price * (1.0 + item["return_upper"] * scale), 2)
                if item["coverage_label"] == "50%":
                    data["price_prediction_interval_50"] = [low_p, high_p]
                elif item["coverage_label"] == "80%":
                    data["price_prediction_interval_80"] = [low_p, high_p]
                elif item["coverage_label"] == "90%":
                    data["price_prediction_interval_90"] = [low_p, high_p]
        except Exception:
            pass

        data["model_version"] = self.model_version
        data["baseline_reference_mae_inr"] = self.baseline_price_mae
        return data
