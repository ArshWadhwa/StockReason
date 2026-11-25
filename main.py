"""StockReason Root Server Entrypoint."""

import sys
import os
from pathlib import Path
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.main import app

if __name__ == "__main__":
    print("=" * 60)
    print(">>> Starting StockReason Intelligence API on http://localhost:8000")
    print(">>> Interactive API Docs available at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
