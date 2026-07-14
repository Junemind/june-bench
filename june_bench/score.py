"""june_bench scorer — the canonical SQuAD/HotpotQA metrics + the three-axis summary (SB0/SB1).

EM and token-F1 are the *same* deterministic metrics Cognee's eval framework reports, so a number
here is directly comparable to published tables (and to June's apex_qa runs — SB1 adds a parity
test against `benchmarks/apex_qa/metrics.py`, the canonical source these mirror). Pure stdlib, so it
runs anywhere with no model.

The summary reports the three axes the suite cares about: **accuracy** (selective EM/F1 over answered
items), **coverage** (fraction answered = 1 − abstention rate), and **cost** (calls / cost per item) —
honest abstention and cost are measured, never free.
"""
from __future__ import annotations

import collections
import re
import string
from collections.abc import Callable, Sequence

from june_bench.ports import Record

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Canonical SQuAD/HotpotQA normalization: lowercase, strip punctuation, drop articles,
    collapse whitespace — so 'The Beatles.' and 'beatles' compare equal."""
    s = (s or "").lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not pred and not g:
        return 1.0
    if not pred or not g:
        return 0.0
    common = collections.Counter(pred) & collections.Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(g)
    return 2 * p * r / (p + r)


def best_over_golds(metric: Callable[[str, str], float], prediction: str,
                    golds: Sequence[str]) -> float:
    return max((metric(prediction, g) for g in golds), default=0.0)


def answer_in_context(context_text: str, golds: Sequence[str]) -> float:
    """Model-free 'did retrieval carry the gold' signal (the gap to EM is what an LLM closes)."""
    hay = normalize_answer(context_text)
    return float(any(normalize_answer(g) and normalize_answer(g) in hay for g in golds))


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def score(records: Sequence[Record]) -> dict:
    """Fold records into the three-axis summary. Accuracy is **selective** (over answered items),
    paired with coverage, so abstaining trades coverage for accuracy in a measured way."""
    n = len(records)
    answered = [r for r in records if not r.abstained]
    em = _mean([best_over_golds(exact_match, r.prediction, r.golds) for r in answered])
    f1 = _mean([best_over_golds(token_f1, r.prediction, r.golds) for r in answered])
    ctx_recall = _mean([answer_in_context(" ".join(r.context), r.golds) for r in records])
    return {
        "n": n,
        "answered": len(answered),
        "abstained": n - len(answered),
        "em": round(em, 4),
        "f1": round(f1, 4),
        "coverage": round(len(answered) / n, 4) if n else 0.0,
        "context_recall": round(ctx_recall, 4),
        "calls_per_answer": round(_mean([float(r.calls) for r in records]), 4),
        "cost_per_answer": round(_mean([float(r.cost) for r in records]), 4),
    }


__all__ = ["normalize_answer", "exact_match", "token_f1", "best_over_golds",
           "answer_in_context", "score"]
