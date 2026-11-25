#!/usr/bin/env python3
"""StockReason — Hugging Face Model Download Utility.

Downloads verified model artifacts from Hugging Face Hub (or local staging cache),
verifies cryptographic checksums, supports pinned revisions, and caches locally.
"""

import os
import sys
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_HF = True
except ImportError:
    HAS_HF = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_models")

DEFAULT_REPO_ID = "stockreason-models"
DEFAULT_REVISION = "main"

# Expected SHA-256 Checksums
EXPECTED_CHECKSUMS = {
    "lightgbm_21d.txt": "6401867a2590147c84fe0e3fc288c461042e9a64ff86b7173aca8123ddd1b916",
    "catboost_21d.cbm": "af8493fe8d15c4a5bef0d005937b5199e57b71e815db984e9510fe31902e150d",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_models(
    dest_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    token: Optional[str] = None,
) -> bool:
    """
    Downloads and verifies model artifacts.
    If HF is unauthenticated or repo is not found, falls back gracefully to local staging/models cache.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN")

    logger.info("Attempting model download for repo: %s (revision: %s)...", repo_id, revision)
    remote_success = False

    if HAS_HF and token:
        try:
            # Download LightGBM
            lgb_path = hf_hub_download(
                repo_id=repo_id,
                filename="lightgbm/model.txt",
                revision=revision,
                token=token,
                local_dir=str(dest_dir),
            )
            shutil.copy(lgb_path, dest_dir / "lightgbm_21d.txt")

            # Download CatBoost
            cb_path = hf_hub_download(
                repo_id=repo_id,
                filename="catboost/model.cbm",
                revision=revision,
                token=token,
                local_dir=str(dest_dir),
            )
            shutil.copy(cb_path, dest_dir / "catboost_21d.cbm")
            remote_success = True
            logger.info("Downloaded remote models from Hugging Face.")
        except Exception as e:
            logger.warning("Remote Hugging Face download failed or repository unreachable: %s", e)

    if not remote_success:
        logger.info("Using local verified artifact cache from models/ directory...")
        for fname in ["lightgbm_21d.txt", "catboost_21d.cbm", "manifest.json", "config.json"]:
            src = Path("models") / fname
            dst = dest_dir / fname
            if src.exists():
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)
            else:
                staging_src = Path("models/hf_staging") / ("lightgbm/model.txt" if "lightgbm" in fname else "catboost/model.cbm" if "catboost" in fname else fname)
                if staging_src.exists() and staging_src.resolve() != dst.resolve():
                    shutil.copy(staging_src, dst)

    # Verify checksums
    all_ok = True
    for fname, expected_hash in EXPECTED_CHECKSUMS.items():
        target = dest_dir / fname
        if target.exists():
            actual_hash = sha256_file(target)
            if actual_hash == expected_hash:
                logger.info("Checksum verified for %s: %s (MATCH)", fname, actual_hash)
            else:
                logger.error("Checksum mismatch for %s: %s != %s", fname, actual_hash, expected_hash)
                all_ok = False
        else:
            logger.warning("Model file %s not found in %s", fname, dest_dir)
            all_ok = False

    return all_ok


def main():
    dest_dir = Path("models")
    ok = download_models(dest_dir)
    if ok:
        print("MODEL_DOWNLOAD_STATUS: SUCCESS")
        sys.exit(0)
    else:
        print("MODEL_DOWNLOAD_STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
