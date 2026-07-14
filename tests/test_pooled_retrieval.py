"""Pooled-corpus retrieval — ingest ALL docs once, retrieve each query over the whole pool.

Tested with a mock transport: pooled mode must (1) ingest the UNION of all examples' docs ONCE
(dedup), (2) NOT create a per-query canvas on each retrieve, (3) map node_ids back to dataset doc-ids
across the whole pool. Contrast with per-query mode (a canvas + ingest per query).
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.ports import Example  # noqa: E402

try:
    import httpx
    _HAVE_HTTPX = True
except Exception:
    _HAVE_HTTPX = False

_NS = uuid.UUID("5e6d0c5e-0000-5000-8000-0000000d0c5e")   # mirrors the engine's _DOCS_NS


def _run(c):
    return asyncio.run(c)


def _mock(calls):
    import httpx as _hx

    def handler(request):
        p = request.url.path
        calls.append((request.method, p))
        if p.endswith("/v1/canvases") and request.method == "POST":
            return _hx.Response(200, json={"canvas_id": "cv-pool"})
        if p.endswith("/v1/ingest/docs"):
            return _hx.Response(200, json={"namespace": str(_NS)})
        if p.endswith("/v1/embeddings/backfill"):
            return _hx.Response(200, json={"embedded": 1})
        if p.endswith("/v1/search"):
            # rank doc "a" (from conv-1) first, then "c" (from conv-2) — cross-conversation
            items = [{"node_id": str(uuid.uuid5(_NS, "a"))},
                     {"node_id": str(uuid.uuid5(_NS, "c"))}]
            return _hx.Response(200, json={"items": items})
        return _hx.Response(200, json={})
    return _hx.MockTransport(handler)


@unittest.skipUnless(_HAVE_HTTPX, "httpx not installed ([june-api] extra)")
class TestPooledRetrieval(unittest.TestCase):
    def _examples(self):
        return [
            Example(qid="q1", question="about a?", golds=("x",),
                    meta={"gold_ids": ["a"], "corpus_docs": [("a", "doc a"), ("b", "doc b")]}),
            Example(qid="q2", question="about c?", golds=("y",),
                    meta={"gold_ids": ["c"], "corpus_docs": [("c", "doc c"), ("a", "doc a")]}),  # 'a' dup
        ]

    def test_pool_ingests_union_once_then_searches_per_query(self):
        from june_bench.systems.june_retrieval import JuneRetrievalSystem
        calls: list = []
        s = JuneRetrievalSystem("http://t", transport=_mock(calls), k=10, pooled=True, backfill=True)
        exs = self._examples()
        n = _run(s.ingest_pool(exs))
        self.assertEqual(n, 3)                                   # a,b,c — 'a' deduped across queries
        # one pool canvas created, docs ingested once, backfill once — BEFORE any query
        self.assertEqual(sum(1 for m, p in calls if p.endswith("/v1/canvases") and m == "POST"), 1)
        self.assertEqual(sum(1 for _, p in calls if p.endswith("/v1/ingest/docs")), 1)
        # now retrieve: NO new canvas/ingest, just a search; maps node_ids → doc_ids across the pool
        calls.clear()
        ranked = _run(s.retrieve(exs[0]))
        self.assertEqual(ranked, ["a", "c"])                    # cross-conversation ranking
        self.assertTrue(all(not p.endswith("/v1/ingest/docs") for _, p in calls))   # no per-query ingest
        self.assertTrue(all(not p.endswith("/v1/canvases") for _, p in calls))      # no per-query canvas
        self.assertTrue(any(p.endswith("/v1/search") for _, p in calls))
        _run(s.aclose())

    def test_run_retrieval_pooled_end_to_end(self):
        from june_bench.systems.june_retrieval import JuneRetrievalSystem, run_retrieval
        calls: list = []
        s = JuneRetrievalSystem("http://t", transport=_mock(calls), k=10, pooled=True)
        rankings, golds = run_retrieval(s, self._examples())
        self.assertEqual(set(rankings), {"q1", "q2"})
        self.assertEqual(rankings["q1"], ["a", "c"])
        self.assertEqual(golds["q1"], ["a"])
        # exactly ONE ingest for the whole run (pooled), not one per query
        self.assertEqual(sum(1 for _, p in calls if p.endswith("/v1/ingest/docs")), 1)

    def test_per_query_mode_ingests_each_query(self):
        # contrast: default (pooled=False) ingests per query
        from june_bench.systems.june_retrieval import JuneRetrievalSystem, run_retrieval
        calls: list = []
        s = JuneRetrievalSystem("http://t", transport=_mock(calls), k=10, pooled=False)
        run_retrieval(s, self._examples())
        self.assertEqual(sum(1 for _, p in calls if p.endswith("/v1/ingest/docs")), 2)   # one per query


if __name__ == "__main__":
    unittest.main()
