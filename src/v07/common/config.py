"""
config.py
---------
Configuration loader and helper for StockReason v0.7.
Loads settings from config/v07.yaml and resolves project-relative paths.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_project_root() -> Path:
    """Returns the absolute path to the repository/project root."""
    return PROJECT_ROOT


def load_v07_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads config/v07.yaml configuration dictionary.
    
    Args:
        config_path: Optional path to YAML config. Defaults to config/v07.yaml in project root.
        
    Returns:
        dict: Parsed configuration dictionary with resolved paths.
    """
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config" / "v07.yaml"
    
    if not path.is_absolute():
        path = root / path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    return cfg
