"""Pooled-QA parity — the anti-drift gate for open-pool answering over HTTP.

The local headline's *open-pool* numbers (`run.py --retrieval pool`) build ONE shared corpus (the
deduped union of every item's passages) and make each question RETRIEVE its gold out of the whole
pool. Online, `JuneApiSystem(pooled=True)` must do the same shape: ingest the union ONCE into a
single shared canvas, then answer each question over that pool WITHOUT any per-question ingest or
canvas. This test pins that contract with a mock transport (no server), mirroring
`test_pooled_retrieval.py` — so the pooled-QA path can't silently regress to record/per-question.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench.ports import Example  # noqa: E402

try:
    import httpx  # noqa: F401 — availability probe for @skipUnless; used via `import httpx as _hx` below
    _HAVE_HTTPX = True
except Exception:
    _HAVE_HTTPX = False


def _run(c):
    return asyncio.run(c)


def _mock(calls):
    import json as _json

    import httpx as _hx

    def handler(request):
        p = request.url.path
        calls.append((request.method, p, request.headers.get("X-Canvas")))
        if p.endswith("/v1/canvases") and request.method == "POST":
            return _hx.Response(200, json={"canvas_id": "cv-pool"})
        if p.endswith("/v1/ingest/docs"):                 # batched pool ingest — log each doc as a DOC row
            try:
                for d in _json.loads(request.content or b"{}").get("docs", []):
                    calls.append(("DOC", str(d.get("id", "")), request.headers.get("X-Canvas")))
            except Exception:  # noqa: BLE001
                pass
            return _hx.Response(200, json={"ingested": 1})
        if p.endswith("/v1/ingest/text"):
            return _hx.Response(200, json={"ok": True})
        if p.endswith("/v1/embeddings/backfill"):
            return _hx.Response(200, json={"embedded": 1})
        if p.endswith("/v1/answer"):
            return _hx.Response(200, json={"answer": "Paris", "mode": "llm_augmented", "degraded": []})
        return _hx.Response(200, json={})
    return _hx.MockTransport(handler)


def _examples():
    return [
        Example(qid="q1", question="capital of France?", golds=("Paris",),
                context=("doc a", "doc b")),
        Example(qid="q2", question="capital of Italy?", golds=("Rome",),
                context=("doc a", "doc c")),   # 'doc a' duplicates q1 → deduped in the pool
    ]


@unittest.skipUnless(_HAVE_HTTPX, "httpx not installed ([june-api] extra)")
class TestPooledQA(unittest.TestCase):
    def _system(self, calls):
        from june_bench.systems.june_api import JuneApiSystem
        return JuneApiSystem("http://t", transport=_mock(calls), isolate=True, backfill=True,
                             pooled=True)

    def test_pool_ingests_union_once_then_answers_over_it(self):
        calls: list = []
        s = self._system(calls)
        exs = _examples()
        n = _run(s.ingest_pool(exs))
        self.assertEqual(n, 3)                                    # a,b,c — 'doc a' deduped across questions
        # exactly ONE pool canvas, docs ingested once (3 unique, via BATCHED /v1/ingest/docs), backfill once
        self.assertEqual(sum(1 for m, p, _ in calls if p.endswith("/v1/canvases") and m == "POST"), 1)
        self.assertEqual(sum(1 for m, _, _ in calls if m == "DOC"), 3)      # 3 unique docs batched in
        self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/ingest/text")), 0)  # NOT per-doc
        self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/embeddings/backfill")), 1)
        self.assertTrue(all(not p.endswith("/v1/answer") for _, p, _ in calls))   # nothing answered yet
        # now answer: NO new canvas, NO new ingest — just a /v1/answer scoped to the SHARED pool canvas
        calls.clear()
        pred = _run(s.answer(exs[0]))
        self.assertEqual(pred.text, "Paris")
        self.assertTrue(pred.meta.get("pooled"))
        self.assertEqual(pred.meta.get("ingest_calls"), 0)
        self.assertTrue(all(not p.endswith("/v1/ingest/text") for _, p, _ in calls))   # no per-Q ingest
        self.assertTrue(all(not p.endswith("/v1/canvases") for _, p, _ in calls))      # no per-Q canvas
        answers = [(p, cv) for m, p, cv in calls if p.endswith("/v1/answer")]
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0][1], "cv-pool")               # answered over the shared pool canvas
        _run(s.aclose())

    def test_ingest_pool_is_idempotent(self):
        calls: list = []
        s = self._system(calls)
        exs = _examples()
        self.assertEqual(_run(s.ingest_pool(exs)), 3)
        self.assertEqual(_run(s.ingest_pool(exs)), 0)            # second call is a no-op (already built)
        self.assertEqual(sum(1 for m, _, _ in calls if m == "DOC"), 3)
        _run(s.aclose())

    def test_runner_pooled_hook_ingests_once_for_whole_run(self):
        from june_bench.runner import run_async
        calls: list = []
        s = self._system(calls)
        records = _run(run_async(s, _examples()))
        self.assertEqual(len(records), 2)
        # the runner's pooled hook ingested the union ONCE for the whole run (3 unique docs), not per-Q
        self.assertEqual(sum(1 for m, _, _ in calls if m == "DOC"), 3)
        self.assertEqual(sum(1 for m, p, _ in calls
                             if p.endswith("/v1/canvases") and m == "POST"), 1)   # pool canvas created once
        self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/answer")), 2)   # one per question
        _run(s.aclose())

    def test_pool_ingest_is_batched_not_per_doc(self):
        # the fix: a big pool uploads via a FEW batched /v1/ingest/docs writes, not 1 write per doc
        # (which lock-storms the endpoint SQLite). Here: 3 docs, 2 batches → 2 POSTs, still 3 docs total.
        import os
        saved = os.environ.get("JUNE_BENCH_POOL_INGEST_BATCHES")
        os.environ["JUNE_BENCH_POOL_INGEST_BATCHES"] = "2"
        try:
            calls: list = []
            s = self._system(calls)
            self.assertEqual(_run(s.ingest_pool(_examples())), 3)
            posts = sum(1 for m, p, _ in calls if m == "POST" and p.endswith("/v1/ingest/docs"))
            self.assertEqual(posts, 2)                          # 3 docs in 2 batched writes
            self.assertEqual(sum(1 for m, _, _ in calls if m == "DOC"), 3)   # all 3 docs still uploaded
            self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/ingest/text")), 0)
            _run(s.aclose())
        finally:
            if saved is None:
                os.environ.pop("JUNE_BENCH_POOL_INGEST_BATCHES", None)
            else:
                os.environ["JUNE_BENCH_POOL_INGEST_BATCHES"] = saved

    def test_batched_ingest_falls_back_to_text_on_404(self):
        # older endpoint without /v1/ingest/docs → 404 → fall back to per-doc /v1/ingest/text (still works)
        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        calls: list = []

        def handler(request):
            p, m = request.url.path, request.method
            calls.append((m, p, request.headers.get("X-Canvas")))
            if p.endswith("/v1/canvases") and m == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-pool"})
            if p.endswith("/v1/ingest/docs"):
                return _hx.Response(404, json={"detail": "not found"})   # no batched endpoint
            return _hx.Response(200, json={})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler), isolate=True,
                          backfill=True, pooled=True)
        self.assertEqual(_run(s.ingest_pool(_examples())), 3)
        self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/ingest/text")), 3)  # per-doc fallback
        _run(s.aclose())

    def test_cleanup_pool_deletes_canvas_and_is_idempotent(self):
        calls: list = []
        s = self._system(calls)
        _run(s.ingest_pool(_examples()))
        calls.clear()
        _run(s.cleanup_pool())
        deletes = [(p, cv) for m, p, cv in calls if m == "DELETE" and p.endswith("/v1/canvases/cv-pool")]
        self.assertEqual(len(deletes), 1)                        # the pool canvas was deleted
        calls.clear()
        _run(s.cleanup_pool())                                   # idempotent — handle cleared, no 2nd DELETE
        self.assertEqual([c for c in calls if c[0] == "DELETE"], [])
        _run(s.aclose())

    def test_runner_deletes_pool_canvas_in_finally(self):
        # the whole point: a pooled run leaves NOTHING behind — the runner deletes the pool canvas on exit
        from june_bench.runner import run_async
        calls: list = []
        s = self._system(calls)
        _run(run_async(s, _examples()))
        self.assertEqual(sum(1 for m, p, _ in calls
                             if m == "DELETE" and p.endswith("/v1/canvases/cv-pool")), 1)
        _run(s.aclose())

    def test_runner_deletes_pool_canvas_even_when_answers_error(self):
        # fail-soft per example must NOT skip cleanup — a run whose answers all error still tears its pool down
        import httpx as _hx

        from june_bench.runner import run_async
        from june_bench.systems.june_api import JuneApiSystem
        calls: list = []

        def handler(request):
            p, m = request.url.path, request.method
            calls.append((m, p, request.headers.get("X-Canvas")))
            if p.endswith("/v1/canvases") and m == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-pool"})
            if p.endswith("/v1/answer"):
                return _hx.Response(500, json={"error": "boom"})     # every answer errors
            return _hx.Response(200, json={})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler),
                          isolate=True, backfill=True, pooled=True)
        _run(run_async(s, _examples()))
        self.assertEqual(sum(1 for m, p, _ in calls
                             if m == "DELETE" and p.endswith("/v1/canvases/cv-pool")), 1)
        _run(s.aclose())

    def test_pool_canvas_registered_before_ingest_so_cleanup_fires_on_ingest_failure(self):
        # REGRESSION: the pool canvas is created before the ingest. If the ingest 500s (the SQLite-lock
        # case that hit real runs), cleanup_pool must STILL delete it. Previously _pool_canvas was set only
        # after a successful ingest, so a failed run leaked its canvas — and leftover pools are exactly what
        # bloat the endpoint's single-writer SQLite into more locks on later runs.
        import os

        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        calls: list = []

        def handler(request):
            p, m = request.url.path, request.method
            calls.append((m, p, request.headers.get("X-Canvas")))
            if p.endswith("/v1/canvases") and m == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-pool"})
            if p.endswith("/v1/ingest/docs") or p.endswith("/v1/ingest/text"):
                return _hx.Response(500, json={"error": "database is locked"})   # ingest fails
            return _hx.Response(200, json={})

        os.environ["JUNE_BENCH_INGEST_RETRIES"] = "1"
        try:
            s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler),
                              isolate=True, backfill=True, pooled=True)
            with self.assertRaises(_hx.HTTPStatusError):
                _run(s.ingest_pool(_examples()))
            self.assertEqual(s._pool_canvas, "cv-pool")     # registered despite failure → cleanup can find it
            calls.clear()
            _run(s.cleanup_pool())
            self.assertEqual(sum(1 for m, p, _ in calls
                                 if m == "DELETE" and p.endswith("/v1/canvases/cv-pool")), 1)
            _run(s.aclose())
        finally:
            os.environ.pop("JUNE_BENCH_INGEST_RETRIES", None)

    def test_sweep_deletes_only_bench_pool_canvases(self):
        # Pre-run sweep must delete leftover `bench-pool` canvases (orphans from earlier runs) but NEVER a
        # user's real canvas. GET /v1/canvases is user-fenced; we filter by the exact pool name.
        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        calls: list = []

        def handler(request):
            p, m = request.url.path, request.method
            calls.append((m, p))
            if p.endswith("/v1/canvases") and m == "GET":
                return _hx.Response(200, json=[
                    {"canvas_id": "cv-old-1", "name": "bench-pool", "created_at": "t1"},
                    {"canvas_id": "cv-mine", "name": "my-real-notes", "created_at": "t2"},
                    {"canvas_id": "cv-old-2", "name": "bench-pool", "created_at": "t3"},
                ])
            return _hx.Response(200, json={"deleted": True})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler), isolate=True, pooled=True)
        swept = _run(s.sweep_stale_pools())
        self.assertEqual(swept, 2)                                    # both bench-pool orphans
        deleted = {p.rsplit("/", 1)[-1] for m, p in calls if m == "DELETE"}
        self.assertEqual(deleted, {"cv-old-1", "cv-old-2"})          # NOT cv-mine (user's real canvas)
        _run(s.aclose())

    def test_sweep_is_failsoft_when_list_route_absent(self):
        # Older endpoint without GET /v1/canvases (404) → sweep is a no-op, never raises, run proceeds.
        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem

        def handler(request):
            if request.method == "GET":
                return _hx.Response(404, json={"detail": "not found"})
            return _hx.Response(200, json={})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler), isolate=True, pooled=True)
        self.assertEqual(_run(s.sweep_stale_pools()), 0)             # no route → nothing swept, no raise
        _run(s.aclose())

    def test_runner_sweeps_before_pool_ingest(self):
        # The runner must sweep BEFORE ingesting the pool, so orphans are cleared before the new write.
        import httpx as _hx

        from june_bench.runner import run_async
        from june_bench.systems.june_api import JuneApiSystem
        order: list = []

        def handler(request):
            p, m = request.url.path, request.method
            if p.endswith("/v1/canvases") and m == "GET":
                order.append("sweep-list")
                return _hx.Response(200, json=[{"canvas_id": "cv-old", "name": "bench-pool",
                                                "created_at": "t"}])
            if p.endswith("/v1/ingest/docs"):
                order.append("ingest")
                return _hx.Response(200, json={})
            if p.endswith("/v1/canvases") and m == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-pool"})
            return _hx.Response(200, json={})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler),
                          isolate=True, backfill=True, pooled=True)
        _run(run_async(s, _examples()))
        self.assertLess(order.index("sweep-list"), order.index("ingest"))   # swept before ingesting
        _run(s.aclose())

    def test_await_ready_proceeds_immediately_when_healthy(self):
        # fast health probe → measured latency < threshold → returns without waiting
        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(lambda r: _hx.Response(200, json={"status": "ok"})),
                          isolate=True, pooled=True)
        _run(s.await_ready())                                    # returns fast, no exception, no hang
        _run(s.aclose())

    def test_await_ready_disabled_by_env_does_not_probe(self):
        import os

        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        hits = {"n": 0}

        def handler(r):
            hits["n"] += 1
            return _hx.Response(200, json={})
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(handler), isolate=True, pooled=True)
        os.environ["JUNE_BENCH_READY_GATE"] = "0"
        try:
            _run(s.await_ready())
            self.assertEqual(hits["n"], 0)                      # gate off → no probe at all
        finally:
            os.environ.pop("JUNE_BENCH_READY_GATE", None)
        _run(s.aclose())

    def test_await_ready_fails_open_when_probe_errors(self):
        # probe 500s → _probe_latency None → proceed immediately (never block the run on a probe failure)
        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(lambda r: _hx.Response(500, json={})),
                          isolate=True, pooled=True)
        self.assertIsNone(_run(s._probe_latency()))
        _run(s.await_ready())                                    # returns, no hang
        _run(s.aclose())

    def test_await_ready_proceeds_after_max_wait_when_still_busy(self):
        # always-busy probe + max_wait=0 → straight to "proceed anyway", bounded, never hangs
        import os

        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(lambda r: _hx.Response(200, json={})),
                          isolate=True, pooled=True)

        async def always_busy(samples=3):
            return 9.0
        s._probe_latency = always_busy                          # over any sane threshold
        os.environ["JUNE_BENCH_READY_MAX_WAIT_S"] = "0"
        try:
            _run(s.await_ready())                               # max_wait 0 → skip loop → proceed, no hang
        finally:
            os.environ.pop("JUNE_BENCH_READY_MAX_WAIT_S", None)
        _run(s.aclose())

    def test_await_ready_waits_then_proceeds_when_busy_then_clears(self):
        # busy first, clears on the next poll → waits one interval then continues
        import os

        import httpx as _hx

        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=_hx.MockTransport(lambda r: _hx.Response(200, json={})),
                          isolate=True, pooled=True)
        seq = iter([9.0, 0.1])                                   # over threshold, then under

        async def fake(samples=3):
            return next(seq)
        s._probe_latency = fake
        os.environ["JUNE_BENCH_READY_INTERVAL_S"] = "1"
        os.environ["JUNE_BENCH_READY_MAX_WAIT_S"] = "5"
        try:
            _run(s.await_ready())                               # 9s>1.5 → wait 1s → 0.1s → proceed
        finally:
            os.environ.pop("JUNE_BENCH_READY_INTERVAL_S", None)
            os.environ.pop("JUNE_BENCH_READY_MAX_WAIT_S", None)
        _run(s.aclose())

    def test_non_pooled_ingests_per_question(self):
        # contrast: default (pooled=False) creates a canvas + ingests per question (record=False path)
        from june_bench.systems.june_api import JuneApiSystem
        calls: list = []
        s = JuneApiSystem("http://t", transport=_mock(calls), isolate=True, pooled=False, record=False)
        _run(s.answer(_examples()[0]))
        self.assertEqual(sum(1 for m, p, _ in calls if p.endswith("/v1/canvases") and m == "POST"), 1)
        self.assertEqual(sum(1 for _, p, _ in calls if p.endswith("/v1/ingest/text")), 2)  # this Q's 2 docs
        _run(s.aclose())


if __name__ == "__main__":
    unittest.main()
