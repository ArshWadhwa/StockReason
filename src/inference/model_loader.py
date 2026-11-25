"""Model Loading & Artifact Registry Module."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import logging
from src.utils.config import MODELS_DIR
from src.models.baseline import NoChangeRandomWalkBaseline, HierarchicalShrinkageBaseline
from src.models.lightgbm_model import LightGBMForecaster
from src.models.catboost_model import CatBoostForecaster

logger = logging.getLogger("model_loader")


class ModelLoader:
    """
    Loads native LightGBM (.txt), CatBoost (.cbm), and statistical baseline models.
    """
    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        self.models_dir = Path(models_dir)

    def load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.models_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return {}

    def load_lightgbm(self, filename: str = "lightgbm_21d.txt") -> Optional[LightGBMForecaster]:
        path = self.models_dir / filename
        if not path.exists():
            # Check fallback in artifacts/v0.9/models/
            legacy_path = Path("artifacts/v0.9/models") / filename
            if legacy_path.exists():
                path = legacy_path
            else:
                logger.warning("LightGBM model not found at %s", path)
                return None

        forecaster = LightGBMForecaster()
        forecaster.load_model(path)
        logger.info("Loaded native LightGBM model from %s", path)
        return forecaster

    def load_catboost(self, filename: str = "catboost_21d.cbm") -> Optional[CatBoostForecaster]:
        path = self.models_dir / filename
        if not path.exists():
            legacy_path = Path("artifacts/v0.9/models") / filename
            if legacy_path.exists():
                path = legacy_path
            else:
                logger.warning("CatBoost model not found at %s", path)
                return None

        forecaster = CatBoostForecaster()
        forecaster.load_model(path)
        logger.info("Loaded native CatBoost model from %s", path)
        return forecaster

    def load_baseline(self, horizon: int = 21) -> NoChangeRandomWalkBaseline:
        return NoChangeRandomWalkBaseline(horizon=horizon)
