"""Retrieval scoring — the IR metrics + turn-grain collapse for june-bench's retrieval mode.

The public harness ships **no June source**, so these standard IR metrics are reimplemented here in
pure stdlib (they mirror `june_ai.retrieval.eval` semantics: binary, multi-gold relevance). A
retrieval run produces, per query, a **ranked list of doc-ids** and a set of **gold doc-ids**; these
fold them into recall@k / nDCG@k / MRR — the same rulers `scripts/retrieval_benchmark.py` reports.

**Turn-grain** (`collapse_turns`): conversational corpora (LoCoMo) are chunked at turn grain
(`<conv>::<turn>`), but gold is scored at the **parent** (session/conversation) grain. Collapsing a
chunk-id ranking to parent-ids — keeping each parent at its *best* (earliest) rank — is the documented
fix that stopped dense recall from collapsing on LoCoMo.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _parent(doc_id: str, sep: str = "::") -> str:
    """The parent (session/conversation) id of a turn-grain chunk id (`<parent>::<turn>` → `<parent>`)."""
    return str(doc_id).split(sep, 1)[0]


def collapse_turns(ranking: Sequence[str], *, sep: str = "::") -> list[str]:
    """Collapse a chunk-id ranking to **parent ids**, keeping each parent at its best (earliest) rank
    (MAX-over-turns). Order preserved; later turns of an already-seen parent are dropped."""
    out: list[str] = []
    seen: set[str] = set()
    for d in ranking:
        p = _parent(d, sep)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def collapse_gold(gold: Sequence[str], *, sep: str = "::") -> set[str]:
    """Gold ids collapsed to parent grain (so a parent counts as found if any of its turns ranked)."""
    return {_parent(g, sep) for g in gold}


def recall_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float:
    """|gold ∩ top-k| / |gold|. 0 when there is no gold (a query with no relevant doc is undefined;
    callers should exclude it — `score_retrieval` does)."""
    if not gold:
        return 0.0
    topk = set(ranking[:k])
    return len(gold & topk) / len(gold)


def ndcg_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG (gain 1 at each relevant rank, discounted by log2(rank+1)) over
    the ideal DCG (all gold ranked first)."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(ranking: Sequence[str], gold: set[str], k: int | None = None) -> float:
    """1 / (rank of the first relevant doc), 0 if none in the top-k (or anywhere when k is None)."""
    if not gold:
        return 0.0
    for i, d in enumerate(ranking if k is None else ranking[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_retrieval(rankings: Mapping[str, Sequence[str]], golds: Mapping[str, Sequence[str]],
                    *, ks: Sequence[int] = (1, 5, 10), turn_grain: bool = False,
                    mrr_k: int | None = 10) -> dict:
    """Fold per-query {qid: ranking} + {qid: gold-ids} into the headline retrieval summary. Only
    queries that **have gold** are scored (judged-but-empty-gold queries are excluded, matching the
    reference harness). ``turn_grain`` collapses both ranking and gold to parent ids first."""
    qids = [q for q in rankings if golds.get(q)]
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    ndcgs: dict[int, list[float]] = {k: [] for k in ks}
    rrs: list[float] = []
    for q in qids:
        ranking = list(rankings[q])
        gold = set(golds[q])
        if turn_grain:
            ranking = collapse_turns(ranking)
            gold = collapse_gold(gold)
        for k in ks:
            recalls[k].append(recall_at_k(ranking, gold, k))
            ndcgs[k].append(ndcg_at_k(ranking, gold, k))
        rrs.append(reciprocal_rank(ranking, gold, mrr_k))
    return {
        "n_queries": len(qids),
        "recall": {k: _mean(recalls[k]) for k in ks},
        "ndcg": {k: _mean(ndcgs[k]) for k in ks},
        "mrr": _mean(rrs),
        "turn_grain": turn_grain,
    }


__all__ = ["collapse_turns", "collapse_gold", "recall_at_k", "ndcg_at_k", "reciprocal_rank",
           "score_retrieval"]
