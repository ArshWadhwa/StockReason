"""
StockReason v0.7 - Module: News Headline Deduplication
Clusters near-duplicate headlines across syndicated media sources using
token-set Jaccard similarity and sequence matching.
"""
import re
import difflib
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd


# Regex patterns to strip trailing publisher attributions
_ATTRIBUTION_PATTERN = re.compile(
    r"\s*([-\|\–\—]\s*([A-Za-z0-9\s\.\&]+)|\[.*?\]|\(.*?\))\s*$"
)

# Common non-informative financial stopwords for title normalization
_TITLE_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "for", "of", "to", "and", "or", "is",
    "are", "was", "were", "by", "from", "with", "as", "about", "into", "through",
    "after", "over", "between", "out", "against", "during", "before", "under",
    "around", "among", "stock", "shares", "share", "today", "latest", "news"
}


def normalize_title(title: str) -> str:
    """
    Normalizes a headline for comparison:
    - Strips publisher attribution suffix (e.g. '- Economic Times', '| Mint')
    - Strips commas between digits (e.g. '18,000' -> '18000')
    - Lowercases text
    - Normalizes financial units (cr/crore, rs/inr)
    - Normalizes whitespace
    """
    if not title:
        return ""

    text = title.strip()

    # 1. Strip publisher suffixes like ' - The Economic Times', ' | Livemint'
    text = _ATTRIBUTION_PATTERN.sub("", text)
    if " - " in text:
        text = text.rsplit(" - ", 1)[0].strip()
    if " | " in text:
        text = text.rsplit(" | ", 1)[0].strip()

    # 2. Lowercase
    text = text.lower()

    # 3. Strip commas between digits (e.g. 18,000 -> 18000)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    # 4. Unit normalizations
    text = re.sub(r"\b(rs\.?|inr|₹)\s*", "rs ", text)
    text = re.sub(r"\b(cr\.?|crores?)\b", "crore", text)
    text = re.sub(r"\b(pct\.?|percent|percentage)\b", "%", text)

    # 5. Remove punctuation except '%'
    text = re.sub(r"[^\w\s%]", " ", text)

    # 6. Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_title_tokens(normalized_title: str, remove_stopwords: bool = True) -> Set[str]:
    """Extracts non-stopword tokens from a normalized title."""
    tokens = set(normalized_title.split())
    if remove_stopwords:
        tokens = {t for t in tokens if t not in _TITLE_STOPWORDS and len(t) > 1}
    return tokens


def title_similarity(title1: str, title2: str) -> float:
    """
    Computes a composite similarity score between two headlines:
    Combines Token Jaccard Similarity (word overlap) and SequenceMatcher ratio (order & phrasing).
    Returns float in [0.0, 1.0].
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 1.0

    # Token Jaccard
    tokens1 = get_title_tokens(norm1)
    tokens2 = get_title_tokens(norm2)

    if not tokens1 or not tokens2:
        jaccard = 0.0
    else:
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0

    # Sequence Matcher
    seq_ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()

    # Balanced score with emphasis on token overlap for syndicated news
    return 0.6 * jaccard + 0.4 * seq_ratio


class DisjointSet:
    """Disjoint Set (Union-Find) with path compression and union by rank."""

    def __init__(self, elements: List[str]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def cluster_headlines(
    items: List[Dict[str, Any]],
    similarity_threshold: float = 0.65,
    title_key: str = "title",
    id_key: str = "event_id",
) -> List[Dict[str, Any]]:
    """
    Clusters a list of headline dictionaries and assigns 'dedup_cluster_id'.
    For large batches with multiple tickers or dates, grouping can be done per ticker.
    """
    if not items:
        return []

    # Assign fallback IDs if missing
    for idx, item in enumerate(items):
        if id_key not in item or not item[id_key]:
            norm = normalize_title(item.get(title_key, ""))
            item[id_key] = hashlib.sha256(f"{norm}_{idx}".encode("utf-8")).hexdigest()[:16]

    ids = [item[id_key] for item in items]
    ds = DisjointSet(ids)

    # Pre-compute normalized titles
    normalized = {item[id_key]: normalize_title(item.get(title_key, "")) for item in items}
    token_cache = {id_val: get_title_tokens(norm) for id_val, norm in normalized.items()}

    n = len(items)
    for i in range(n):
        id_i = ids[i]
        tokens_i = token_cache[id_i]
        norm_i = normalized[id_i]

        for j in range(i + 1, n):
            id_j = ids[j]
            tokens_j = token_cache[id_j]

            # Fast token overlap pre-filter: must share at least 2 tokens (or 1 if short)
            common = len(tokens_i & tokens_j)
            if common < min(len(tokens_i), len(tokens_j), 2):
                continue

            sim = title_similarity(norm_i, normalized[id_j])
            if sim >= similarity_threshold:
                ds.union(id_i, id_j)

    # Map root IDs to clean deterministic cluster IDs
    cluster_mapping = {}
    for id_val in ids:
        root = ds.find(id_val)
        if root not in cluster_mapping:
            cluster_mapping[root] = f"cluster_{root[:12]}"

    for item in items:
        root = ds.find(item[id_key])
        item["dedup_cluster_id"] = cluster_mapping[root]

    return items


def deduplicate_dataframe(
    df: pd.DataFrame,
    similarity_threshold: float = 0.65,
    title_col: str = "title",
    id_col: str = "event_id",
    keep: str = "earliest",
) -> pd.DataFrame:
    """
    Clusters duplicate headlines in a DataFrame and optionally deduplicates it.
    If keep == 'all', returns the original DataFrame with 'dedup_cluster_id' attached.
    If keep == 'earliest', keeps the earliest article per cluster.
    If keep == 'highest_confidence', keeps the row with highest confidence.
    """
    if df.empty:
        df["dedup_cluster_id"] = []
        return df

    records = df.to_dict(orient="records")
    clustered = cluster_headlines(
        records,
        similarity_threshold=similarity_threshold,
        title_key=title_col,
        id_key=id_col,
    )
    result_df = pd.DataFrame(clustered)

    if keep == "all":
        return result_df

    if keep == "earliest":
        # Sort by available_at or published_at if available
        sort_cols = [c for c in ["available_at", "published_at_utc"] if c in result_df.columns]
        if sort_cols:
            result_df = result_df.sort_values(sort_cols[0], ascending=True)
        return result_df.drop_duplicates(subset=["dedup_cluster_id"], keep="first").reset_index(drop=True)

    if keep == "highest_confidence" and "confidence" in result_df.columns:
        result_df = result_df.sort_values("confidence", ascending=False)
        return result_df.drop_duplicates(subset=["dedup_cluster_id"], keep="first").reset_index(drop=True)

    return result_df.drop_duplicates(subset=["dedup_cluster_id"], keep="first").reset_index(drop=True)
