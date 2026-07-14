"""JuneRetrievalSystem — id-preserving ingest → search → node→doc mapping.

A MockTransport unit test (no server) pins the request shapes + the node_id↔doc_id mapping; an
optional in-process integration test runs the real `serve_local` app (sparse/FTS lane, no network)
end to end through the retrieval scorer.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.ports import Example  # noqa: E402
from june_bench.retrieval import score_retrieval  # noqa: E402

try:
    import httpx
    _HAVE_HTTPX = True
except Exception:
    _HAVE_HTTPX = False


def _run(c):
    return asyncio.run(c)


@unittest.skipUnless(_HAVE_HTTPX, "httpx not installed ([june-api] extra)")
class TestRetrievalMapping(unittest.TestCase):
    def test_ingest_search_maps_node_ids_back_to_doc_ids(self):
        ns = uuid.UUID("5e6d0c5e-0000-5000-8000-0000000d0c5e")
        # the mock ranks the two corpus docs by their deterministic node_ids (reverse order)
        d1, d2 = "conv::s1", "conv::s2"
        n1, n2 = str(uuid.uuid5(ns, d1)), str(uuid.uuid5(ns, d2))

        def handler(request):
            p = request.url.path
            if p.endswith("/v1/canvases") and request.method == "POST":
                return httpx.Response(200, json={"canvas_id": "cv1"})
            if p.endswith("/v1/ingest/docs"):
                return httpx.Response(200, json={"nodes_written": 2, "namespace": str(ns)})
            if p.endswith("/v1/search"):
                return httpx.Response(200, json={"items": [{"node_id": n2}, {"node_id": n1},
                                                           {"node_id": str(uuid.uuid4())}]})  # a stray id
            return httpx.Response(200, json={})
        from june_bench.systems.june_retrieval import JuneRetrievalSystem
        sysm = JuneRetrievalSystem("http://t", transport=httpx.MockTransport(handler), k=10)
        ex = Example(qid="conv", question="q?", golds=(),
                     meta={"gold_ids": [d1], "corpus_docs": [(d1, "text one"), (d2, "text two")]})
        ranked = _run(sysm.retrieve(ex))
        self.assertEqual(ranked, [d2, d1])           # mapped to doc-ids, stray id dropped, order kept
        s = score_retrieval({ex.qid: ranked}, {ex.qid: [d1]}, ks=(1, 2))
        self.assertEqual(s["recall"][1], 0.0)        # d1 at rank 2
        self.assertEqual(s["recall"][2], 1.0)


@unittest.skipUnless(_HAVE_HTTPX, "httpx")
class TestRetrievalInProcess(unittest.TestCase):
    """End-to-end against the real serve_local app over an in-process ASGI transport (sparse lane)."""
    def test_smoke_locomo_recall(self):
        import importlib
        import os
        os.environ.setdefault("JUNE_BENCH_DB", "/tmp/jb_ret_test.db")
        os.environ.setdefault("JUNE_API_KEYS", "local:00000000-0000-0000-0000-000000000001:admin")
        try:
            serve_local = importlib.import_module("deploy.serve_local")
        except Exception as exc:  # noqa: BLE001 — only runs where the engine src is importable
            self.skipTest(f"serve_local not importable here: {exc}")
        from june_bench.datasets import registry
        from june_bench.systems.june_retrieval import JuneRetrievalSystem, run_retrieval
        app = serve_local.build_app()
        sysm = JuneRetrievalSystem("http://test", api_key="local", k=5,
                                   transport=httpx.ASGITransport(app=app))
        exs = list(registry.get("locomo").load("smoke"))
        rankings, golds = run_retrieval(sysm, exs)
        s = score_retrieval(rankings, golds, ks=(1, 5))
        self.assertEqual(s["n_queries"], 2)
        self.assertGreaterEqual(s["recall"][5], 0.5)   # the gold session is retrievable by FTS


if __name__ == "__main__":
    unittest.main()
