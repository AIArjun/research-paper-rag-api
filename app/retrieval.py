"""Bounded hybrid ranking and page-local evidence windows; no provider calls."""

import math
import re
from collections import Counter


STOP_WORDS = frozenset(
    "a an the in on at to of for and or is are was were does do how what which why "
    "as by with from it its their this that used use paper research".split()
)
CANDIDATE_LIMIT = 20


def terms(text: str) -> list[str]:
    # A small plural normalization keeps identifiers and numbers intact.
    return [
        word[:-1] if len(word) > 4 and word.endswith("s") else word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def lexical_rank(question: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """BM25 on the already filtered, capacity-limited in-process corpus."""
    words = [Counter(terms(chunk["text"])) for chunk in chunks]
    lengths = [sum(word.values()) for word in words]
    average = sum(lengths) / max(1, len(words)) or 1
    frequencies = Counter(term for word in words for term in word)
    query = set(terms(question))
    scores = []
    for chunk, word, length in zip(chunks, words, lengths):
        score = sum(
            math.log(1 + (len(words) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            * word[term] * 2.5 / (word[term] + 1.5 * (0.25 + 0.75 * length / average))
            for term in query if word[term]
        )
        if score > 0:
            scores.append((chunk, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]["chunk_id"]))[:CANDIDATE_LIMIT]


def hybrid_rank(dense: list[tuple[dict, float]], lexical: list[tuple[dict, float]],
                top_k: int) -> list[tuple[dict, float]]:
    """Reciprocal rank fusion. Scores express ranking, never confidence."""
    scores, chunks = {}, {}
    for ranked in (dense, lexical):
        seen = set()
        for rank, (chunk, _) in enumerate(ranked[:CANDIDATE_LIMIT], 1):
            key = chunk["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [(chunks[key], scores[key]) for key in keys]


def evidence_window(chunk: dict, page_text: str, max_chars: int) -> str:
    """Expand around a seed's offsets without crossing its physical PDF page."""
    if max_chars <= 0:
        return ""
    start, end = chunk.get("char_start"), chunk.get("char_end")
    if not page_text or type(start) is not int or type(end) is not int:
        return chunk["text"][:max_chars]
    if not 0 <= start < end <= len(page_text):
        return chunk["text"][:max_chars]
    offset = max(0, min((start + end - max_chars) // 2, len(page_text) - max_chars))
    return page_text[offset:offset + max_chars]
