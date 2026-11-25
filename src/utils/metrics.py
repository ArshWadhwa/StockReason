"""General Utility & Mathematical Helpers."""

import hashlib
import os
from pathlib import Path
from typing import Union


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file efficiently in 8MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()
