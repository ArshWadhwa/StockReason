import json
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MOCK_DIR = BASE_DIR / "data" / "mock"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class DataService:
    """
    Data abstraction layer for Module 4.
    In Weeks 1-3: Reads structured mock JSON matching M1/M2/M3 agreed schemas.
    In Week 4: Automatically drops in real parquet/model files when available,
    meaning zero controller or UI changes are needed.
    """

    def __init__(self, force_mock: bool = True):
        self.force_mock = force_mock
        self._load_caches()

    def _load_caches(self):
        prices_file = MOCK_DIR / "prices_mock.json"
        sentiment_file = MOCK_DIR / "sentiment_mock.json"
        predictions_file = MOCK_DIR / "predictions_mock.json"
        signals_file = MOCK_DIR / "signals_mock.json"
        backtest_file = MOCK_DIR / "backtest_mock.json"

        self.prices = json.loads(prices_file.read_text()) if prices_file.exists() else {}
        self.sentiment = json.loads(sentiment_file.read_text()) if sentiment_file.exists() else {}
        self.predictions = json.loads(predictions_file.read_text()) if predictions_file.exists() else {}
        self.signals = json.loads(signals_file.read_text()) if signals_file.exists() else {}
        self.backtest = json.loads(backtest_file.read_text()) if backtest_file.exists() else {}

        # M2 Integration: Incorporate live M2 processed sentiment if available
        self._load_m2_sentiment()
        # M1 Integration: Incorporate live M1 processed price features if available
        self._load_m1_prices()

    def _load_m2_sentiment(self):
        m2_features_path = PROCESSED_DIR / "sentiment_features.parquet"
        if m2_features_path.exists():
            try:
                import pandas as pd
                df_sent = pd.read_parquet(m2_features_path)
                for ticker, grp in df_sent.groupby("ticker"):
                    if ticker in self.sentiment:
                        latest = grp.sort_values(by="date").iloc[-1]
                        self.sentiment[ticker]["avg_sentiment_score"] = float(latest.get("sentiment_mean", self.sentiment[ticker]["avg_sentiment_score"]))
                        div_flag = int(latest.get("sentiment_divergence_flag", 0))
                        div_desc = str(latest.get("divergence_desc", ""))
                        self.sentiment[ticker]["divergence_alert"] = {
                            "has_divergence": div_flag != 0,
                            "message": div_desc or ("Normal: Sentiment aligned" if div_flag == 0 else "Divergence detected")
                        }
            except Exception:
                pass

    def _load_m1_prices(self):
        m1_price_path = PROCESSED_DIR / "price_features.parquet"
        if m1_price_path.exists():
            try:
                import pandas as pd
                df_prices = pd.read_parquet(m1_price_path)
                for ticker, grp in df_prices.groupby("ticker"):
                    grp_sorted = grp.sort_values(by="date")
                    if ticker in self.prices and not grp_sorted.empty:
                        latest_row = grp_sorted.iloc[-1]
                        self.prices[ticker]["current_price"] = float(latest_row.get("close", self.prices[ticker]["current_price"]))
                        if len(grp_sorted) >= 2:
                            prev_close = float(grp_sorted.iloc[-2].get("close", latest_row.get("close")))
                            self.prices[ticker]["day_change_pct"] = round(((latest_row.get("close") - prev_close) / prev_close) * 100, 2)
            except Exception:
                pass

    def get_universes(self) -> Dict[str, Any]:
        indices = [
            {"symbol": "^NSEI", "name": "NIFTY 50", "category": "broad", "base_level": 22200.0},
            {"symbol": "^NSEBANK", "name": "NIFTY BANK", "category": "sectoral", "base_level": 47300.0},
            {"symbol": "^CNXIT", "name": "NIFTY IT", "category": "sectoral", "base_level": 37800.0},
            {"symbol": "^NSENEXT50", "name": "NIFTY NEXT 50", "category": "holdout", "base_level": 60500.0},
            {"symbol": "^INDIAVIX", "name": "India VIX", "category": "volatility", "base_level": 14.2}
        ]
        stocks = []
        for t, data in self.prices.items():
            stocks.append({
                "ticker": t,
                "name": data["name"],
                "sector": data["sector"],
                "universe": data["universe"],
                "current_price": data["current_price"],
                "day_change_pct": data["day_change_pct"]
            })
        return {"indices": indices, "stocks": stocks}

    def get_market_overview(self) -> Dict[str, Any]:
        return {
            "indices": [
                {"symbol": "^NSEI", "name": "NIFTY 50", "level": 22212.5, "change_pct": 0.72, "status": "bullish"},
                {"symbol": "^NSEBANK", "name": "NIFTY BANK", "level": 47320.0, "change_pct": 0.45, "status": "neutral"},
                {"symbol": "^CNXIT", "name": "NIFTY IT", "level": 37850.2, "change_pct": 1.25, "status": "bullish"},
                {"symbol": "^NSENEXT50", "name": "NIFTY NEXT 50", "level": 60520.1, "change_pct": 0.38, "status": "neutral"},
                {"symbol": "^INDIAVIX", "name": "India VIX", "level": 13.85, "change_pct": -4.2, "status": "low_vol"}
            ],
            "macro": [
                {"name": "USD/INR", "value": 82.88, "change_pct": -0.05},
                {"name": "Brent Crude Oil", "value": 83.15, "change_pct": -0.85}
            ],
            "market_regime": "low_vol_trending",
            "active_coverage_count": len(self.prices)
        }

    def get_stock_prices(self, ticker: str, days: int = 180) -> Optional[Dict[str, Any]]:
        stock = self.prices.get(ticker)
        if not stock:
            return None
        res = dict(stock)
        res["history"] = stock["history"][-days:]
        return res

    def get_stock_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.sentiment.get(ticker)

    def get_stock_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.predictions.get(ticker)

    def get_stock_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.signals.get(ticker)

    def get_ranked_signals(self) -> List[Dict[str, Any]]:
        sig_list = list(self.signals.values())
        sig_list.sort(key=lambda x: (
            {"BUY": 3, "HOLD": 2, "AVOID": 1}.get(x["signal"], 0),
            x["signal_score"]
        ), reverse=True)
        return sig_list

    def get_backtest(self) -> Dict[str, Any]:
        return self.backtest

    def get_daily_digest(self) -> Dict[str, Any]:
        ranked = self.get_ranked_signals()
        buys = [s["ticker"] for s in ranked if s["signal"] == "BUY"]
        avoids = [s["ticker"] for s in ranked if s["signal"] == "AVOID"]

        return {
            "digest_date": "2024-03-01",
            "market_summary": (
                "Nifty 50 advanced +0.72% while India VIX decreased -4.2% to 13.85, "
                "reinforcing the Low-Volatility Trending regime. Strong breadth observed across IT and Banking."
            ),
            "signal_upgrades": [
                {
                    "ticker": "INFY.NS",
                    "previous_signal": "HOLD",
                    "new_signal": "BUY",
                    "primary_catalyst": "FinBERT sentiment surged into bullish territory (+0.64) backed by strategic IT deals."
                },
                {
                    "ticker": "RELIANCE.NS",
                    "previous_signal": "HOLD",
                    "new_signal": "BUY",
                    "primary_catalyst": "MC Dropout confidence rose to 88% as price cleared 50-day moving average."
                }
            ],
            "signal_downgrades": [
                {
                    "ticker": "PNB.NS",
                    "previous_signal": "HOLD",
                    "new_signal": "AVOID",
                    "primary_catalyst": "Model disagreement between LSTM (+1.2%) and XGBoost (-5.3%) plus adverse sentiment tone."
                }
            ],
            "top_opportunities": buys[:3],
            "high_risk_alerts": avoids[:2]
        }
