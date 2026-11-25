#!/usr/bin/env python3
"""StockReason — Automated Environment & Workspace Setup Utility.

Verifies Python environment, dependencies, dataset integrity, and model artifacts.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import CANONICAL_DATASET_PATH, CANONICAL_SHA256, MODELS_DIR
from src.utils.metrics import compute_file_sha256
from scripts.download_models import download_models


def main():
    print("=" * 60)
    print("StockReason — Workspace Verification & Setup")
    print("=" * 60)

    # 1. Python version
    py_ver = sys.version_info
    print(f"[+] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 10):
        print("[-] Error: Python 3.10+ required.")
        sys.exit(1)

    # 2. Dependencies
    required_packages = ["pandas", "pyarrow", "numpy", "scipy", "sklearn", "lightgbm", "catboost", "fastapi", "uvicorn"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[-] Missing required packages: {missing}")
        print("    Run: pip install -r requirements.txt")
    else:
        print("[+] Core dependencies verified.")

    # 3. Models
    print("[+] Checking model artifacts...")
    download_models(MODELS_DIR)

    # 4. Dataset
    print(f"[+] Checking canonical dataset at: {CANONICAL_DATASET_PATH}")
    if CANONICAL_DATASET_PATH.exists():
        actual_hash = compute_file_sha256(CANONICAL_DATASET_PATH)
        if actual_hash == CANONICAL_SHA256:
            print(f"[+] Canonical dataset SHA-256 verified: {actual_hash[:16]}... (PASS)")
        else:
            print(f"[!] Warning: Dataset hash mismatch: {actual_hash} != {CANONICAL_SHA256}")
    else:
        print(f"[!] Warning: Canonical dataset not found at {CANONICAL_DATASET_PATH}.")
        print("    Teammates can run model-only inference (Workflow A) without full historical dataset.")

    print("=" * 60)
    print("StockReason setup verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
