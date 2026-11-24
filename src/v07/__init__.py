"""StockReason v0.7 trustworthy forecasting and evaluation engine package."""
import sys
from pathlib import Path

_ws_root = str(Path(__file__).resolve().parents[2])
_src_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
    sys.path.insert(0, _ws_root)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

