"""
StockReason - Module 2: Baseline Sentiment Scorer (TextBlob)
Provides rapid rule-based sentiment polarity scoring as a lightweight baseline
and offline fallback.
"""
from typing import List, Dict, Any
from textblob import TextBlob


def score_single_text_baseline(text: str) -> Dict[str, Any]:
    """
    Computes baseline sentiment metrics using TextBlob.
    Returns polarity (-1.0 to 1.0), subjectivity (0.0 to 1.0), categorical label, and confidence.
    """
    if not text or not text.strip():
        return {
            "score_textblob": 0.0,
            "subjectivity": 0.0,
            "sentiment_label": "neutral",
            "confidence": 1.0
        }
        
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    
    # Label thresholding
    if polarity > 0.05:
        label = "positive"
        confidence = min(1.0, 0.5 + abs(polarity) / 2.0)
    elif polarity < -0.05:
        label = "negative"
        confidence = min(1.0, 0.5 + abs(polarity) / 2.0)
    else:
        label = "neutral"
        confidence = max(0.5, 1.0 - abs(polarity))
        
    return {
        "score_textblob": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "sentiment_label": label,
        "confidence": round(confidence, 4)
    }


def score_headlines_baseline(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scores a list of article dictionaries using the baseline TextBlob scorer.
    Adds baseline fields to each article.
    """
    scored_articles = []
    for art in articles:
        text = art.get("title", "")
        if art.get("summary"):
            text = f"{text}. {art.get('summary')}"
            
        scores = score_single_text_baseline(text)
        art_copy = dict(art)
        art_copy.update({
            "score_textblob": scores["score_textblob"],
            "score_finbert": scores["score_textblob"],  # initial baseline fallback
            "sentiment_label": scores["sentiment_label"],
            "confidence": scores["confidence"]
        })
        scored_articles.append(art_copy)
        
    return scored_articles
