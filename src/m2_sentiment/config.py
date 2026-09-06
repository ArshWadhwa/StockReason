"""
StockReason - Module 2 (News & Sentiment Track) Configuration
Defines ticker aliases, query terms, paths, and model settings.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure target directories exist
RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Company name / alias lookup for rich Indian financial news retrieval
TICKER_QUERY_MAP = {
    # NIFTY 50 & Major Stocks
    "RELIANCE.NS": ["Reliance Industries", "RIL", "Mukesh Ambani Reliance", "Jio Financial"],
    "TCS.NS": ["Tata Consultancy Services", "TCS", "Tata Consultancy"],
    "INFY.NS": ["Infosys", "Infosys quarterly results", "Salil Parekh"],
    "HDFCBANK.NS": ["HDFC Bank", "HDFC Bank merger", "HDFC Q3"],
    "ICICIBANK.NS": ["ICICI Bank", "Sandeep Bakhshi ICICI"],
    "BHARTIARTL.NS": ["Bharti Airtel", "Airtel 5G", "Sunil Mittal Airtel"],
    "ITC.NS": ["ITC Limited", "ITC cigarette hotel demerger"],
    "LT.NS": ["Larsen & Toubro", "L&T infrastructure order"],
    "SBIN.NS": ["State Bank of India", "SBI", "State Bank quarterly"],
    "AXISBANK.NS": ["Axis Bank", "Axis Bank earnings"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank", "Uday Kotak"],
    "INDUSINDBK.NS": ["IndusInd Bank", "IndusInd earnings"],
    "BANKBARODA.NS": ["Bank of Baroda", "BoB"],
    "PNB.NS": ["Punjab National Bank", "PNB"],
    "AUBANK.NS": ["AU Small Finance Bank", "AU Bank"],
    "IDFCFIRSTB.NS": ["IDFC First Bank", "V Vaidyanathan IDFC"],
    "WIPRO.NS": ["Wipro", "Wipro guidance", "Wipro CEO"],
    "HCLTECH.NS": ["HCL Technologies", "HCL Tech"],
    "TECHM.NS": ["Tech Mahindra", "Tech M"],
    "TATAMOTORS.NS": ["Tata Motors", "Tata Motors EV", "JLR Tata"],
    "MARUTI.NS": ["Maruti Suzuki", "Maruti sales"],
    "SUNPHARMA.NS": ["Sun Pharmaceutical", "Sun Pharma"],
    "CIPLA.NS": ["Cipla", "Cipla USFDA"],
    "TATASTEEL.NS": ["Tata Steel", "Tata Steel Europe"],
    "JSWSTEEL.NS": ["JSW Steel", "Sajjan Jindal JSW"],
    
    # Indices & Macro
    "^NSEI": ["NIFTY 50", "Nifty index", "Indian stock market", "NSE Nifty"],
    "^NSEBANK": ["Nifty Bank", "Bank Nifty", "Indian banking sector"],
    "^INDIAVIX": ["India VIX", "market volatility India", "NSE volatility index"]
}

# Supported NLP Models
FINBERT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_HEADLINES_PER_TICKER = 25
