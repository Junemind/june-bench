"""SB3 · JuneApiSystem against a mock June server (no real endpoint, no June source).

Uses `httpx.MockTransport` to stand in for `/v1/answer` + `/v1/ingest/text`, so the request shape and
response parsing are proven without a server. `skipUnless` httpx is installed (the `[june-api]` extra).
Also checks the `june-local` stub errors clearly and the registry exposes the systems.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import run, score                  # noqa: E402
from june_bench import systems                      # noqa: E402
from june_bench.datasets import registry            # noqa: E402
from june_bench.ports import Example                # noqa: E402

try:
    import httpx
    _HAVE_HTTPX = True
except Exception:
    _HAVE_HTTPX = False


def _run(c):
    return asyncio.run(c)


def _mock(answer="Paris", degraded=None, record=None):
    import httpx as _hx

    def handler(request: "_hx.Request") -> "_hx.Response":
        if record is not None:
            record.append((request.method, request.url.path))
        # isolation lifecycle: the client creates a canvas first and uses the server-issued id.
        if request.url.path.endswith("/v1/canvases") and request.method == "POST":
            return _hx.Response(200, json={"canvas_id": "cv-1", "name": "", "created_at": ""})
        if request.url.path.endswith("/v1/ingest/text"):
            return _hx.Response(200, json={"ok": True})
        return _hx.Response(200, json={"answer": answer, "citations": [], "used_edge_ids": [],
                                       "degraded": degraded or [], "mode": "llm_augmented"})
    return _hx.MockTransport(handler)


@unittest.skipUnless(_HAVE_HTTPX, "httpx not installed ([june-api] extra)")
class TestJuneApiSystem(unittest.TestCase):
    def _sys(self, **kw):
        from june_bench.systems.june_api import JuneApiSystem
        return JuneApiSystem("http://june.test", transport=_mock(**kw))

    def test_answer_parses_answerout(self):
        pred = _run(self._sys(answer="Paris").answer(Example(qid="1", question="capital of France?",
                                                             golds=("Paris",))))
        self.assertEqual(pred.text, "Paris")
        self.assertFalse(pred.meta["abstained"])
        self.assertEqual(pred.meta["calls"], 1)

    def test_abstention_detected_from_degraded(self):
        pred = _run(self._sys(answer="", degraded=["abstain:insufficient_evidence"])
                    .answer(Example(qid="1", question="q", golds=("x",))))
        self.assertTrue(pred.meta["abstained"])

    def test_abstention_detected_from_llm_sentinel(self):
        # the LLM synthesizer abstains by saying "I don't know" (no degraded marker) — must still
        # count as abstained, not as an answered-but-wrong item.
        for phrase in ("I don't know", "I don't know.", "  i do not know  "):
            pred = _run(self._sys(answer=phrase).answer(Example(qid="1", question="q", golds=("x",))))
            self.assertTrue(pred.meta["abstained"], phrase)

    def test_real_answer_not_marked_abstained(self):
        pred = _run(self._sys(answer="Paris").answer(Example(qid="1", question="q", golds=("Paris",))))
        self.assertFalse(pred.meta["abstained"])

    def test_ingest_is_called_for_corpus_then_answer(self):
        calls: list = []
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_mock(answer="Business Administration",
                                                                 record=calls))
        ex = list(registry.get("locomo").load("smoke"))   # memory-QA → has corpus
        recs = _run(_run_async(sysm, ex))
        paths = [p for _, p in calls]
        self.assertTrue(any(p.endswith("/v1/ingest/text") for p in paths))   # ingested
        self.assertTrue(any(p.endswith("/v1/answer") for p in paths))        # then answered
        self.assertEqual(len(recs), 2)

    def test_context_fed_and_canvas_isolated(self):
        # context-QA passages must reach June (it answers over its graph), and every request for
        # an example must be scoped to that example's own canvas (no cross-question leakage).
        seen: list = []
        import httpx as _hx

        creates: list = []

        def handler(request):
            if request.url.path.endswith("/v1/canvases") and request.method == "POST":
                creates.append(request.headers.get("x-canvas", ""))      # create runs in home ws (no canvas)
                return _hx.Response(200, json={"canvas_id": "cv-q7"})
            seen.append((request.url.path, request.headers.get("x-canvas", "")))
            if request.url.path.endswith("/v1/ingest/text"):
                return _hx.Response(200, json={"ok": True})
            return _hx.Response(200, json={"answer": "X", "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler))
        ex = Example(qid="q7", question="who?", golds=("x",),
                     context=("passage A", "passage B"), corpus=("doc 1",))
        _run(sysm.answer(ex))
        ingests = [p for p, _ in seen if p.endswith("/v1/ingest/text")]
        self.assertEqual(len(ingests), 3)                       # 2 context passages + 1 corpus doc fed in
        self.assertEqual(creates, [""])                         # one canvas created, in the home workspace
        self.assertTrue(all(cv == "cv-q7" for _, cv in seen))   # ingest + answer all in the server-issued canvas

    def test_backfill_called_after_ingest_in_canvas(self):
        # with backfill=True the dense lane must be primed: /v1/embeddings/backfill fires AFTER the
        # ingests and BEFORE the answer, all in the question's canvas.
        seen: list = []
        import httpx as _hx

        def handler(request):
            if request.url.path.endswith("/v1/canvases") and request.method == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-q9"})
            seen.append((request.url.path, request.headers.get("x-canvas", "")))
            return _hx.Response(200, json={"ok": True, "canvas_id": "cv-1", "answer": "X",
                                           "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler), backfill=True)
        ex = Example(qid="q9", question="who?", golds=("x",), corpus=("doc 1",))
        _run(sysm.answer(ex))
        paths = [p for p, _ in seen]
        self.assertIn("/v1/embeddings/backfill", paths)
        self.assertLess(paths.index("/v1/ingest/text"), paths.index("/v1/embeddings/backfill"))
        self.assertLess(paths.index("/v1/embeddings/backfill"), paths.index("/v1/answer"))
        self.assertTrue(all(cv == "cv-q9" for _, cv in seen))

    def test_backfill_failsoft_on_model_free_endpoint(self):
        # a model-free endpoint 400s on backfill ('no embedder configured'); the answer must still land.
        import httpx as _hx

        def handler(request):
            if request.url.path.endswith("/v1/embeddings/backfill"):
                return _hx.Response(400, json={"detail": "no embedder configured"})
            if request.url.path.endswith("/v1/canvases") and request.method == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-q1"})
            return _hx.Response(200, json={"ok": True, "answer": "Paris", "degraded": [], "mode": "local"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler), backfill=True)
        pred = _run(sysm.answer(Example(qid="q1", question="capital?", golds=("Paris",), corpus=("d",))))
        self.assertEqual(pred.text, "Paris")          # backfill 400 did not break the answer

    def test_cleanup_deletes_canvas_after_answer(self):
        # Q5: with cleanup=True the question's canvas is DELETEd after the answer (best-effort).
        seen: list = []
        import httpx as _hx

        def handler(request):
            seen.append((request.method, request.url.path))
            if request.url.path.endswith("/v1/canvases") and request.method == "POST":
                return _hx.Response(200, json={"canvas_id": "cv-q3"})
            return _hx.Response(200, json={"ok": True, "answer": "X", "degraded": [], "mode": "llm",
                                           "nodes_deleted": 0, "edges_deleted": 0, "deleted": True})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler), cleanup=True)
        _run(sysm.answer(Example(qid="q3", question="q", golds=("x",), corpus=("d",))))
        self.assertIn(("DELETE", "/v1/canvases/cv-q3"), seen)
        # delete happens AFTER the answer
        methods = [m for m, _ in seen]
        self.assertLess(methods.index("POST"), len(methods) - 1 - methods[::-1].index("DELETE"))

    def test_ingest_calls_recorded_in_meta(self):
        # Q4: ingest-side work is reported separately from the cost-axis `calls` (=1 answer call).
        pred = _run(self._sys(answer="A").answer(
            Example(qid="1", question="q", golds=("x",), corpus=("d1", "d2"), context=("c1",))))
        self.assertEqual(pred.meta["calls"], 1)               # the answer-LLM call (cost axis)
        self.assertEqual(pred.meta["ingest_calls"], 3)        # 2 corpus + 1 context ingested
        self.assertIn("backfilled", pred.meta)

    def test_record_mode_sends_passages_and_skips_ingest(self):
        # record=True → no ingest, no canvas; the example's context+corpus ride inline on /v1/answer
        # as `passages` (the answerer-isolating headline setting).
        import json as _json

        import httpx as _hx
        seen: list = []
        captured: dict = {}

        def handler(request):
            seen.append(request.url.path)
            if request.url.path.endswith("/v1/answer"):
                captured.update(_json.loads(request.content.decode() or "{}"))
            return _hx.Response(200, json={"answer": "Paris", "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler), record=True)
        ex = Example(qid="q1", question="where?", golds=("Paris",),
                     context=("The Eiffel Tower is in Paris.",), corpus=("a doc",))
        pred = _run(sysm.answer(ex))
        self.assertTrue(all(not p.endswith("/v1/ingest/text") for p in seen))   # no ingest
        self.assertTrue(all(not p.endswith("/v1/canvases") for p in seen))      # no canvas lifecycle
        self.assertIn("/v1/answer", seen)
        self.assertEqual(captured.get("passages"), ["The Eiffel Tower is in Paris.", "a doc"])
        self.assertEqual(pred.text, "Paris")
        self.assertEqual(pred.meta["ingest_calls"], 0)

    def test_multihop_param_rides_on_answer_payload(self):
        # params={"multihop": True} → the /v1/answer body carries multihop so the route takes the
        # decompose→merge path (the A3 multi-hop number). Rides through record mode too.
        import json as _json

        import httpx as _hx
        captured: dict = {}

        def handler(request):
            if request.url.path.endswith("/v1/answer"):
                captured.update(_json.loads(request.content.decode() or "{}"))
            return _hx.Response(200, json={"answer": "X", "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler),
                             record=True, params={"multihop": True, "max_subqueries": 3})
        _run(sysm.answer(Example(qid="q1", question="q", golds=("x",), context=("p",))))
        self.assertTrue(captured.get("multihop"))
        self.assertEqual(captured.get("max_subqueries"), 3)

    _PROXY_KEYS = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                   "HTTPS_PROXY", "https_proxy")

    def _from_env_with(self, **env):
        # Build via from_env with a controlled env; neutralize any ambient proxy (a CI/sandbox SOCKS
        # proxy would otherwise make the default httpx transport raise) — proxy config is not under test.
        import os

        from june_bench.systems.june_api import from_env
        keys = ("JUNE_BENCH_JUNE_URL", "JUNE_BENCH_JUNE_MULTIHOP",
                "JUNE_BENCH_JUNE_MAX_SUBQ", *self._PROXY_KEYS)
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in self._PROXY_KEYS:
                os.environ.pop(k, None)
            os.environ["JUNE_BENCH_JUNE_URL"] = "http://june.test"
            for k in ("JUNE_BENCH_JUNE_MULTIHOP", "JUNE_BENCH_JUNE_MAX_SUBQ"):
                os.environ.pop(k, None)
            os.environ.update(env)
            return from_env()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_from_env_reads_multihop_flag(self):
        # JUNE_BENCH_JUNE_MULTIHOP=1 (+ MAX_SUBQ) → from_env injects them as AnswerIn overrides.
        sysm = self._from_env_with(JUNE_BENCH_JUNE_MULTIHOP="1", JUNE_BENCH_JUNE_MAX_SUBQ="5")
        self.assertTrue(sysm._params.get("multihop"))
        self.assertEqual(sysm._params.get("max_subqueries"), 5)

    def test_from_env_no_multihop_by_default(self):
        sysm = self._from_env_with()
        self.assertNotIn("multihop", sysm._params)

    def test_no_cleanup_by_default(self):
        seen: list = []
        import httpx as _hx

        def handler(request):
            seen.append(request.method)
            return _hx.Response(200, json={"ok": True, "canvas_id": "cv-1", "answer": "X",
                                           "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler))  # cleanup off
        _run(sysm.answer(Example(qid="q1", question="q", golds=("x",), corpus=("d",))))
        self.assertNotIn("DELETE", seen)

    def test_byo_llm_key_sent_as_header(self):
        # BYO-key: the caller's LLM key rides on X-LLM-Key so the endpoint synthesizes on the CALLER's dime.
        seen: list = []
        import httpx as _hx

        def handler(request):
            seen.append(request.headers.get("x-llm-key"))
            return _hx.Response(200, json={"ok": True, "canvas_id": "cv-1", "answer": "X",
                                           "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler), llm_key="sk-caller")
        _run(sysm.answer(Example(qid="q1", question="q", golds=("x",), corpus=("d",))))
        self.assertTrue(all(k == "sk-caller" for k in seen))   # every request carried the caller key

    def test_no_llm_key_header_when_unset(self):
        seen: list = []
        import httpx as _hx

        def handler(request):
            seen.append("x-llm-key" in request.headers)
            return _hx.Response(200, json={"ok": True, "canvas_id": "cv-1", "answer": "X",
                                           "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler))  # no llm_key
        _run(sysm.answer(Example(qid="q1", question="q", golds=("x",), corpus=("d",))))
        self.assertFalse(any(seen))

    def test_no_backfill_by_default(self):
        seen: list = []
        import httpx as _hx

        def handler(request):
            seen.append(request.url.path)
            return _hx.Response(200, json={"ok": True, "canvas_id": "cv-1", "answer": "X",
                                           "degraded": [], "mode": "llm"})
        from june_bench.systems.june_api import JuneApiSystem
        sysm = JuneApiSystem("http://june.test", transport=_hx.MockTransport(handler))  # backfill off
        _run(sysm.answer(Example(qid="q1", question="q", golds=("x",), corpus=("d",))))
        self.assertNotIn("/v1/embeddings/backfill", seen)

    def test_end_to_end_score_over_mock(self):
        # mock returns the gold for each smoke Q → EM 1.0 through the real runner+scorer
        sysm = self._sys(answer="7 May 2023")
        # only the first locomo smoke gold is "7 May 2023"; use a per-question oracle mock instead:
        from june_bench.systems.june_api import JuneApiSystem

        def handler(request):
            import json as _json
            if request.url.path.endswith("/v1/canvases") and request.method == "POST":
                return httpx.Response(200, json={"canvas_id": "cv-1"})
            body = _json.loads(request.content.decode() or "{}")
            q = body.get("query", "")
            ans = "7 May 2023" if "LGBTQ" in q else "Business Administration"
            return httpx.Response(200, json={"answer": ans, "citations": [], "used_edge_ids": [],
                                             "degraded": [], "mode": "llm"})
        sysm = JuneApiSystem("http://june.test", transport=httpx.MockTransport(handler))
        s = score(run(sysm, registry.get("locomo"), split="smoke"))
        self.assertEqual(s["em"], 1.0)


async def _run_async(system, examples):
    from june_bench.runner import run_async
    return await run_async(system, examples)


class TestRegistryAndLocalStub(unittest.TestCase):
    def test_systems_registered(self):
        for name in ("june-api", "june", "june-local"):
            self.assertIn(name, systems.names())

    def test_june_local_errors_clearly(self):
        with self.assertRaises((RuntimeError, NotImplementedError)):
            systems.get("june-local")

    def test_june_api_without_url_errors_clearly(self):
        import os
        old = os.environ.pop("JUNE_BENCH_JUNE_URL", None)
        try:
            with self.assertRaises(ValueError):
                systems.get("june-api")
        finally:
            if old is not None:
                os.environ["JUNE_BENCH_JUNE_URL"] = old


if __name__ == "__main__":
    unittest.main()
