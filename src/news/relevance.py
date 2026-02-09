from __future__ import annotations

import re

# Words to strip from search queries (too generic / noise)
_STOPWORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "been",
    "by", "in", "on", "at", "to", "for", "of", "with", "from", "as",
    "this", "that", "it", "its", "or", "and", "not", "no", "yes",
    "do", "does", "did", "has", "have", "had", "can", "could", "would",
    "should", "may", "might", "shall", "before", "after", "during",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "if", "than", "then", "about", "into", "through", "above", "below",
    "between", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "any", "only",
}


def extract_search_queries(question: str, max_queries: int = 3) -> list[str]:
    """Turn a market question into useful news search queries.

    Returns up to *max_queries* queries derived from the question text.
    """
    # Clean question marks and leading "Will" / "Is" etc.
    cleaned = re.sub(r"[?!.,;:\"']", "", question).strip()

    # Full question (minus punctuation) is always the first query
    queries: list[str] = [cleaned]

    # Extract named entities / key phrases (crude heuristic: capitalised runs)
    entities = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
    for entity in entities:
        if entity.lower() not in _STOPWORDS and len(entity) > 2:
            queries.append(entity)

    # Keyword query — remove stopwords
    words = cleaned.split()
    keywords = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    if keywords:
        queries.append(" ".join(keywords[:6]))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:max_queries]
