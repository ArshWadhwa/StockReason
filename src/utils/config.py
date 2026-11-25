"""StockReason Configuration & Path Management."""

from pathlib import Path
import os

# Project Root is defined relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Core Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CLEANED_DIR = DATA_DIR / "cleaned"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

# Canonical Dataset Defaults
DEFAULT_CANONICAL_DATASET = DATA_PROCESSED_DIR / "canonical_panel.parquet"
LEGACY_CANONICAL_DATASET = DATA_DIR / "v0.13b" / "canonical_panel_v013b.parquet"

CANONICAL_DATASET_PATH = DEFAULT_CANONICAL_DATASET if DEFAULT_CANONICAL_DATASET.exists() else LEGACY_CANONICAL_DATASET
CANONICAL_SHA256 = "bd717815bd559aff58ed0ac9ef5ad69e4bf06dd3051c709d55036ac866305f24"

# Standard Horizons
HORIZONS = [1, 5, 21, 63, 126]
PRIMARY_HORIZON = 21
