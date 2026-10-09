"""Reference-based and reference-free metrics for LLM / RAG answers."""
from __future__ import annotations

import re
from collections import Counter

_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be",
         "must", "at", "any", "by", "it", "this", "that", "with", "before", "after"}

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\bnot (?:mentioned|stated|provided|available|in the context)\b",
    r"\bcannot (?:answer|find|determine)\b",
    r"\bno information\b",
]


def normalize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def exact_match(answer: str, reference: str) -> float:
    return float(normalize(answer) == normalize(reference))


def token_f1(answer: str, reference: str) -> float:
    """SQuAD-style token-overlap F1."""
    a, r = Counter(normalize(answer)), Counter(normalize(reference))
    overlap = sum((a & r).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Share of required phrases present in the answer (case-insensitive substring)."""
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the retrieved context.

    A cheap, deterministic hallucination signal: low values mean the answer
    introduces facts the context does not support.
    """
    words = normalize(answer)
    if not words:
        return 1.0
    ctx = set(normalize(context))
    return sum(w in ctx for w in words) / len(words)


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)
