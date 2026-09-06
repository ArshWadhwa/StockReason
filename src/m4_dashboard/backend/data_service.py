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
        # Keep full list for home page ticker tape preview
        self.all_signals = dict(self.signals)
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
                real_tickers = set(df_prices["ticker"].unique())
                
                # Filter prices and signals to only the real stocks
                filtered_prices = {}
                for ticker, grp in df_prices.groupby("ticker"):
                    grp_sorted = grp.sort_values(by="date")
                    if not grp_sorted.empty:
                        latest_row = grp_sorted.iloc[-1]
                        prev_close = float(grp_sorted.iloc[-2].get("close", latest_row.get("close"))) if len(grp_sorted) >= 2 else float(latest_row.get("close"))
                        curr_close = float(latest_row.get("close"))
                        day_chg = round(((curr_close - prev_close) / prev_close) * 100, 2)
                        
                        # Build full real history list for charting
                        history = []
                        for _, row in grp_sorted.iterrows():
                            dt_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                            history.append({
                                "date": dt_str,
                                "open": float(row.get("open", row["close"])),
                                "high": float(row.get("high", row["close"])),
                                "low": float(row.get("low", row["close"])),
                                "close": float(row["close"]),
                                "volume": int(row.get("volume", 0)),
                                "sma_20": float(row.get("sma_20", row["close"])) if "sma_20" in row and not pd.isna(row["sma_20"]) else None,
                                "sma_50": float(row.get("sma_50", row["close"])) if "sma_50" in row and not pd.isna(row["sma_50"]) else None,
                                "rsi_14": float(row.get("rsi_14", 50.0)) if "rsi_14" in row and not pd.isna(row["rsi_14"]) else None,
                                "macd": float(row.get("macd_line_12_26", 0.0)) if "macd_line_12_26" in row and not pd.isna(row.get("macd_line_12_26")) else None,
                                "regime": str(row.get("market_regime", "trending")).lower()
                            })
                        
                        stock_meta = self.prices.get(ticker, {
                            "name": "Reliance Industries Ltd",
                            "sector": "Energy",
                            "universe": "NIFTY 50"
                        })
                        
                        filtered_prices[ticker] = {
                            "ticker": ticker,
                            "name": stock_meta.get("name", ticker),
                            "sector": stock_meta.get("sector", "Large Cap"),
                            "universe": stock_meta.get("universe", "NIFTY 50"),
                            "current_price": curr_close,
                            "day_change_pct": day_chg,
                            "history": history
                        }

                if filtered_prices:
                    self.prices = filtered_prices
                    self.signals = {t: sig for t, sig in self.signals.items() if t in real_tickers}
                    self.sentiment = {t: sent for t, sent in self.sentiment.items() if t in real_tickers}
                    self.predictions = {t: pred for t, pred in self.predictions.items() if t in real_tickers}
            except Exception as e:
                print("Error loading m1 prices:", e)

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

    def get_ranked_signals(self, include_all: bool = False) -> List[Dict[str, Any]]:
        source = self.all_signals if include_all else self.signals
        sig_list = list(source.values())
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
