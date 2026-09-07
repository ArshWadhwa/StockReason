"""
StockReason Root Entrypoint
Allows running the backend API directly via:
    python main.py
or
    uvicorn main:app --reload
"""
import sys
import os
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.m4_dashboard.backend.main import app

if __name__ == "__main__":
    print("=" * 60)
    print(">>> Starting StockReason Intelligence API on http://localhost:8000")
    print(">>> Interactive API Docs available at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
