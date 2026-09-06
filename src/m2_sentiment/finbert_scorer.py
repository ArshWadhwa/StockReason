"""
StockReason - Module 2: FinBERT Financial NLP Sentiment Engine
Uses Hugging Face Transformers with the ProsusAI/finbert model
to extract domain-specific financial sentiment from headlines.
Includes safe lazy imports and graceful fallback to baseline/lexicon scoring.
"""
from typing import List, Dict, Any, Optional
from .config import FINBERT_MODEL_NAME, DEFAULT_BATCH_SIZE
from .baseline_scorer import score_headlines_baseline, score_single_text_baseline

_tokenizer = None
_model = None
_torch = None
_device = "cpu"
_finbert_available = None


# Curated financial keywords for high-accuracy financial scoring fallback
FINANCIAL_KEYWORD_WEIGHTS = {
    # Positive triggers
    "surges": 0.6, "jumps": 0.5, "beats": 0.5, "soars": 0.6, "profit": 0.4,
    "revenue up": 0.5, "growth": 0.3, "dividend": 0.3, "expansion": 0.3,
    "bullish": 0.6, "outperform": 0.5, "record high": 0.6, "buyback": 0.4,
    "order win": 0.4, "approved": 0.3, "gain": 0.3, "upgrade": 0.5,
    # Negative triggers
    "plunges": -0.6, "slumps": -0.5, "falls": -0.4, "drops": -0.4, "losses": -0.6,
    "penalty": -0.6, "lawsuit": -0.5, "fraud": -0.8, "cuts": -0.4, "warning": -0.5,
    "downgrade": -0.5, "bearish": -0.6, "investigation": -0.6, "default": -0.7,
    "probe": -0.5, "scam": -0.8, "decline": -0.4, "recession": -0.6
}


def _financial_lexicon_score(text: str) -> float:
    """Enhances baseline score with domain financial dictionary weights."""
    t_lower = text.lower()
    adjustment = 0.0
    for word, weight in FINANCIAL_KEYWORD_WEIGHTS.items():
        if word in t_lower:
            adjustment += weight
    return max(-1.0, min(1.0, adjustment))


def load_finbert(model_name: str = FINBERT_MODEL_NAME):
    """
    Safe lazy loader for FinBERT model and tokenizer.
    Ensures safe initialization and caches in memory.
    """
    global _tokenizer, _model, _torch, _device, _finbert_available
    
    if _finbert_available is False:
        return None, None
        
    if _tokenizer is None or _model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _torch = torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _model.to(_device)
            _model.eval()
            _finbert_available = True
        except Exception as e:
            print(f"[FinBERT Notice] FinBERT transformer unavailable ({e}). Using financial lexicon scorer.")
            _finbert_available = False
            return None, None
            
    return _tokenizer, _model


def score_texts_finbert(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Dict[str, Any]]:
    """
    Runs batch inference on a list of financial texts using FinBERT.
    Falls back gracefully to financial lexicon + TextBlob scoring if FinBERT is unavailable.
    """
    if not texts:
        return []
        
    tokenizer, model = load_finbert()
    
    # Fallback to enhanced financial lexicon scorer
    if tokenizer is None or model is None:
        results = []
        for text in texts:
            base = score_single_text_baseline(text)
            lex_adj = _financial_lexicon_score(text)
            
            # Combine TextBlob with financial dictionary
            combined_score = base["score_textblob"] * 0.4 + lex_adj * 0.6
            combined_score = round(max(-1.0, min(1.0, combined_score)), 4)
            
            if combined_score > 0.10:
                lbl = "positive"
                conf = min(1.0, 0.6 + abs(combined_score) * 0.4)
                probs = {"positive": conf, "neutral": round(1.0 - conf, 4), "negative": 0.0}
            elif combined_score < -0.10:
                lbl = "negative"
                conf = min(1.0, 0.6 + abs(combined_score) * 0.4)
                probs = {"positive": 0.0, "neutral": round(1.0 - conf, 4), "negative": conf}
            else:
                lbl = "neutral"
                conf = 0.8
                probs = {"positive": 0.1, "neutral": 0.8, "negative": 0.1}
                
            results.append({
                "score_finbert": combined_score,
                "sentiment_label": lbl,
                "confidence": round(conf, 4),
                "probabilities": probs
            })
        return results

    results: List[Dict[str, Any]] = []
    id2label = model.config.id2label

    for i in range(0, len(texts), batch_size):
        batch_texts = [t[:512] if t else "Neutral financial update" for t in texts[i : i + batch_size]]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(_device)
        
        with _torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = _torch.softmax(logits, dim=-1).cpu().numpy()
            
        for prob_vec in probs:
            prob_dict = {id2label[idx].lower(): float(prob_vec[idx]) for idx in range(len(prob_vec))}
            pos_p = prob_dict.get("positive", 0.0)
            neg_p = prob_dict.get("negative", 0.0)
            neu_p = prob_dict.get("neutral", 0.0)
            
            # Compound financial score
            compound_score = pos_p - neg_p
            
            # Dominant label
            max_label = max(prob_dict, key=prob_dict.get)
            confidence = float(prob_dict[max_label])
            
            results.append({
                "score_finbert": round(compound_score, 4),
                "sentiment_label": max_label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "positive": round(pos_p, 4),
                    "neutral": round(neu_p, 4),
                    "negative": round(neg_p, 4)
                }
            })
            
    return results


def score_headlines(
    articles: List[Dict[str, Any]],
    use_finbert: bool = True
) -> List[Dict[str, Any]]:
    """
    Scores a list of article dictionaries with FinBERT (or enhanced financial scorer).
    """
    if not articles:
        return []
        
    scored = score_headlines_baseline(articles)
    texts = [art.get("title", "") for art in scored]
    finbert_results = score_texts_finbert(texts)
    
    for i, res in enumerate(finbert_results):
        scored[i]["score_finbert"] = res["score_finbert"]
        scored[i]["sentiment_label"] = res["sentiment_label"]
        scored[i]["confidence"] = res["confidence"]
        scored[i]["finbert_probs"] = res.get("probabilities", {})
        
    return scored
