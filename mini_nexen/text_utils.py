from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    return _SENTENCE_RE.split(text)


def score_documents(query_tokens: Iterable[str], documents: list[tuple[str, str]]) -> list[tuple[int, float]]:
    """
    Score documents with a light-weight TF-IDF style heuristic.

    documents: list of (doc_id, text)
    returns: list of (index, score)
    """
    tokens = [t for t in query_tokens if t]
    if not tokens:
        return [(idx, 0.0) for idx in range(len(documents))]

    doc_term_counts = []
    doc_lengths = []
    term_doc_freq = Counter()

    for _, text in documents:
        doc_tokens = tokenize(text)
        counts = Counter(doc_tokens)
        doc_term_counts.append(counts)
        doc_lengths.append(max(1, len(doc_tokens)))
        for term in set(tokens):
            if term in counts:
                term_doc_freq[term] += 1

    total_docs = max(1, len(documents))
    scores = []
    for idx, counts in enumerate(doc_term_counts):
        score = 0.0
        for term in tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = term_doc_freq.get(term, 1)
            idf = math.log((total_docs + 1) / (df + 0.5))
            score += (1 + math.log(tf)) * idf
        score /= doc_lengths[idx]
        scores.append((idx, score))

    return scores


def top_sentences(text: str, keywords: Iterable[str], limit: int = 3) -> list[str]:
    terms = [t.lower() for t in keywords if t]
    if not terms:
        return []

    sentences = split_sentences(text)
    scored = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(sentence_lower.count(term) for term in terms)
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [sentence for _, sentence in scored[:limit]]
