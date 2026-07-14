"""SB4 · CogneeSystem adapter (with fakes — no cognee, no keys).

The cognee machinery is injected, so the adapter contract is provable without cognee: it ingests
the corpus (memory-QA) or the per-question passages (context-QA) and parses the answer. The real
`from_cognee` factory raises clearly when cognee isn't installed.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import run, score                     # noqa: E402
from june_bench import systems                          # noqa: E402
from june_bench.datasets import registry                # noqa: E402
from june_bench.ports import Example                     # noqa: E402
from june_bench.systems.cognee import CogneeSystem       # noqa: E402


def _run(c):
    return asyncio.run(c)


class _Fake:
    def __init__(self, answer="Paris"):
        self.ingested: list = []
        self._answer = answer

    def answer_fn(self, question):
        return self._answer

    def ingest_fn(self, docs):
        self.ingested.extend(docs)


class TestCogneeAdapter(unittest.TestCase):
    def test_answer_maps_to_prediction(self):
        f = _Fake("Paris")
        pred = _run(CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn).answer(
            Example(qid="1", question="capital of France?", golds=("Paris",))))
        self.assertEqual(pred.text, "Paris")
        self.assertFalse(pred.meta["abstained"])
        self.assertEqual(pred.meta["engine"], "cognee_graph_completion")

    def test_empty_answer_abstains(self):
        f = _Fake("")
        pred = _run(CogneeSystem(f.answer_fn).answer(Example(qid="1", question="q", golds=("x",))))
        self.assertTrue(pred.meta["abstained"])

    def test_memory_qa_ingests_corpus(self):
        f = _Fake("Business Administration")
        ex = list(registry.get("locomo").load("smoke"))
        recs = _run(_run_async(CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn), ex))
        self.assertTrue(f.ingested)                       # corpus was ingested
        self.assertEqual(len(recs), 2)

    def test_context_qa_ingests_passages(self):
        f = _Fake("American")
        ex = list(registry.get("hotpot").load("smoke"))[:1]
        _run(CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn).answer(ex[0]))
        self.assertTrue(f.ingested)                       # the question's passages were ingested

    def test_end_to_end_score(self):
        # fake returns the gold per question → EM 1.0 through the real runner+scorer
        def answer_fn(q):
            return "7 May 2023" if "LGBTQ" in q else "Business Administration"
        s = score(run(CogneeSystem(answer_fn), registry.get("locomo"), split="smoke"))
        self.assertEqual(s["em"], 1.0)


class TestCogneePooled(unittest.TestCase):
    """OPEN-POOL parity: pooled Cognee must build the KG ONCE over the deduped union (like the local
    driver + JuneApiSystem), NOT prune+rebuild per question — otherwise a June-pool vs Cognee run is an
    unmatched retrieval task."""

    def test_ingest_pool_builds_once_and_dedups(self):
        f = _Fake("x")
        s = CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn, pooled=True)
        exs = [Example(qid="1", question="q1", golds=("a",), context=("P1", "SHARED")),
               Example(qid="2", question="q2", golds=("b",), context=("P2", "SHARED"))]
        n = _run(s.ingest_pool(exs))
        self.assertEqual(n, 3)                                # P1, SHARED, P2 — SHARED deduped once
        self.assertEqual(sorted(f.ingested), ["P1", "P2", "SHARED"])
        _run(s.ingest_pool(exs))                              # idempotent — no second build
        self.assertEqual(sorted(f.ingested), ["P1", "P2", "SHARED"])

    def test_pooled_answer_does_not_reingest(self):
        f = _Fake("Paris")
        s = CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn, pooled=True)
        _run(s.ingest_pool([Example(qid="1", question="q", golds=("a",), context=("P1",))]))
        f.ingested.clear()
        _run(s.answer(Example(qid="1", question="q", golds=("a",), context=("P1", "P2"))))
        self.assertEqual(f.ingested, [])                      # pooled → NO per-question ingest/prune

    def test_pooled_ingest_is_noop(self):
        # runner's per-example ingest(ex.corpus) must not touch the pooled graph
        f = _Fake("x")
        s = CogneeSystem(f.answer_fn, ingest_fn=f.ingest_fn, pooled=True)
        _run(s.ingest(["some", "corpus"]))
        self.assertEqual(f.ingested, [])

    def test_runner_hook_pools_via_ingest_pool(self):
        # end-to-end through the real runner: pooled system exposes ingest_pool + _pooled, so the runner
        # ingests the union ONCE before the loop (matched to June-pool). EM 1.0 with a gold-returning fake.
        f_ing: list = []

        def ingest_fn(docs):
            f_ing.extend(docs)

        def answer_fn(q):
            return "American"
        ex = list(registry.get("hotpot").load("smoke"))
        s = CogneeSystem(answer_fn, ingest_fn=ingest_fn, pooled=True)
        recs = _run(_run_async(s, ex))
        self.assertTrue(f_ing)                                # the pool was built once via the hook
        self.assertEqual(len(recs), len(ex))
        self.assertTrue(all(r.meta.get("pooled") for r in recs))


class TestPoolFlagParity(unittest.TestCase):
    """Anti-drift guard: the pool flag must pool June and Cognee IDENTICALLY, or an H2H is unmatched.
    Locks that Cognee reads the SAME `JUNE_BENCH_JUNE_POOL` June-api reads, plus the `COGNEE_POOL`
    override — no cognee/httpx install needed (tests the pure env decision)."""

    def setUp(self):
        import os
        self._saved = {k: os.environ.get(k) for k in ("JUNE_BENCH_JUNE_POOL", "COGNEE_POOL")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        import os
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _pool(self):
        from june_bench.systems.cognee import _pool_from_env
        return _pool_from_env()

    def test_same_flag_pools_both_sides(self):
        import os
        os.environ["JUNE_BENCH_JUNE_POOL"] = "1"      # the ONE flag an H2H sets for June
        self.assertTrue(self._pool())                 # → Cognee pools too (matched task)

    def test_unset_is_not_pooled(self):
        self.assertFalse(self._pool())

    def test_cognee_pool_override_opts_out(self):
        import os
        os.environ["JUNE_BENCH_JUNE_POOL"] = "1"
        os.environ["COGNEE_POOL"] = "0"               # keep June pooled, opt Cognee out
        self.assertFalse(self._pool())

    def test_cognee_pool_override_opts_in(self):
        import os
        os.environ["COGNEE_POOL"] = "yes"
        self.assertTrue(self._pool())

    def test_reads_the_same_env_var_june_api_reads(self):
        # if June-api ever renames its pool env, this fails → forces Cognee to be updated in lockstep
        import inspect

        from june_bench.systems import june_api
        self.assertIn("JUNE_BENCH_JUNE_POOL", inspect.getsource(june_api.from_env))


class TestTerseGraphCompletion(unittest.TestCase):
    """The prompt-wiring that makes Cognee answer in bare spans (fair vs strict EM). Tested with a fake
    `cognee` (no install): the PRIMARY path passes `system_prompt_path=answer_simple_question_benchmark
    .txt` — Cognee's own shipped benchmark prompt, identical to its eval_framework (the local 0.417)."""

    def test_primary_uses_shipped_benchmark_prompt_file(self):
        from june_bench.systems.cognee import terse_graph_completion
        calls = []

        class _FakeCognee:
            async def search(self, **kw):
                calls.append(kw)
                return ["paris"]

        out = _run(terse_graph_completion(_FakeCognee(), "GRAPH_COMPLETION", "capital?",
                                          prompt_path="answer_simple_question_benchmark.txt",
                                          inline_prompt="terse"))
        self.assertEqual(out, ["paris"])
        self.assertEqual(len(calls), 1)                                    # primary path hit, no fallback
        self.assertEqual(calls[0]["system_prompt_path"], "answer_simple_question_benchmark.txt")
        self.assertNotIn("system_prompt", calls[0])                        # not the verbose default

    def test_falls_back_to_inline_then_bare_on_old_cognee(self):
        from june_bench.systems.cognee import terse_graph_completion
        calls = []

        class _OldCognee:
            async def search(self, **kw):
                # old cognee: rejects BOTH prompt kwargs (TypeError), accepts only the bare call
                if "system_prompt_path" in kw or "system_prompt" in kw:
                    raise TypeError("unexpected kwarg")
                calls.append(kw)
                return ["x"]

        out = _run(terse_graph_completion(_OldCognee(), "GC", "q",
                                          prompt_path="bench.txt", inline_prompt="terse"))
        self.assertEqual(out, ["x"])
        self.assertEqual(calls, [{"query_type": "GC", "query_text": "q"}])  # degraded to bare call

    def test_env_overrides_prompt_path(self):
        # from_cognee reads COGNEE_SYSTEM_PROMPT_PATH for the prompt file (verified indirectly: the
        # helper just forwards whatever prompt_path it's given).
        from june_bench.systems.cognee import terse_graph_completion
        seen = {}

        class _FakeCognee:
            async def search(self, **kw):
                seen.update(kw)
                return ["y"]

        _run(terse_graph_completion(_FakeCognee(), "GC", "q",
                                    prompt_path="custom_prompt.txt", inline_prompt="t"))
        self.assertEqual(seen["system_prompt_path"], "custom_prompt.txt")


async def _run_async(system, examples):
    from june_bench.runner import run_async
    return await run_async(system, examples)


class TestRegistryAndFactory(unittest.TestCase):
    def test_cognee_registered(self):
        self.assertIn("cognee", systems.names())

    def test_factory_errors_without_cognee(self):
        try:
            import cognee  # noqa: F401
        except Exception:
            cognee_installed = False
        else:
            cognee_installed = True
        # skipTest() raises SkipTest (a subclass of Exception), so it must live OUTSIDE the import
        # try/except — otherwise the bare `except` swallows the skip and the error-path assertion
        # runs even when cognee IS installed (no RuntimeError → false failure).
        if cognee_installed:
            self.skipTest("cognee is installed; factory error path not exercised")
        with self.assertRaises(RuntimeError):
            systems.get("cognee")


if __name__ == "__main__":
    unittest.main()
