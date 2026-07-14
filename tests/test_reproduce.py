"""Tests for the `reproduce` command — preset wiring, input resolution, and the MOAT-LEAK guard.

The moat rule (`moat-no-model-names`): no user-facing surface — the friendly output OR `--show-config`
— may name the commodity retrieval models. Even the endpoint's health response reports capabilities
(lanes on/off), never a model id. This gate scans the reproduce module so a future edit can't leak a
model name into the shipped package.

NB: the banned tokens are base64-encoded below so this guard file itself carries no readable model id
(it must survive being read in the sdist without leaking the very names it guards).
"""
from __future__ import annotations

import base64
import pathlib
import types
import unittest

from june_bench import reproduce

# Internal model families that must not appear in the friendly output / module source. Encoded so the
# guard ships no readable model id; decoded at runtime for the scan.
#
# `bge` / `bge-large` are DELIBERATELY NOT on this list. bge-large-en-v1.5 is June's dense-lane embedder
# AND a *published benchmark parameter* — a stranger must know it to run a same-embedder H2H, so it is
# disclosed in README.md + the fairness methodology and is allowed to appear as the default embedder in
# source. The moat is the cost-gated extraction/graph pipeline and its internal model choices, NOT the
# commodity dense model — so those internal families stay banned (encoded above) and tested here.
_BANNED = tuple(
    base64.b64decode(b).decode()
    for b in ("bm9taWM=", "Z2xpbmVy", "bWluaWxt", "ZTUt", "aW5zdHJ1Y3Rvci0=")
)


class TestNoMoatLeak(unittest.TestCase):
    def test_how_line_names_no_models(self):
        low = reproduce._HOW_LINE.lower()
        for term in _BANNED:
            self.assertNotIn(term, low, f"friendly _HOW_LINE leaks a model name: {term!r}")
        # it should still be MEANINGFUL — describe the capability
        self.assertIn("open-pool", low)
        self.assertIn("retrieval", low)

    def test_module_source_hardcodes_no_model_names(self):
        # The whole module: capabilities only. Model ids must come from the live health probe, not
        # literals — so no banned term appears anywhere in the source.
        src = pathlib.Path(reproduce.__file__).read_text(encoding="utf-8").lower()
        for term in _BANNED:
            self.assertNotIn(term, src, f"reproduce.py hardcodes a model name: {term!r}")

    def test_no_model_name_anywhere_in_shipped_package(self):
        # Generalized guard (§10): scan EVERY shipped .py in the package — not just reproduce.py — so a
        # model name can't hide in systems/ or an error string (the `EMBED=<internal-id>` class of leak).
        # Match on word boundaries so dataset-y substrings (e.g. "economic" contains a banned token) don't trip it.
        import re

        import june_bench
        pkg = pathlib.Path(june_bench.__file__).parent
        patterns = [(t, re.compile(rf"\b{re.escape(t)}", re.IGNORECASE)) for t in _BANNED]
        for py in pkg.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for term, pat in patterns:
                self.assertIsNone(
                    pat.search(text),
                    f"{py.relative_to(pkg)} leaks a model name: {term!r}")


class TestIngestBatchRecommendation(unittest.TestCase):
    """The batch recommendation is COMPUTED (dataset size ÷ a ~30-question batch) and scales with N —
    so a bigger run recommends more batches (avoids the endpoint's SQLite lock). The ~30 cap is
    empirical: 50q (≈491 deduped passages) in ONE write hit the lock, so 50q must split (rec ≥ 2)."""

    def test_recommendation_scales_with_dataset_size(self):
        r = reproduce._recommended_batches
        self.assertEqual(r(30), 1)          # small run → one batch, no prompt
        self.assertEqual(r(50), 2)          # 50q must split — one write of ~491 passages locked the DB
        self.assertEqual(r(100), 4)         # matches the known-good manual 4-batch n=100 run
        self.assertEqual(r(200), 7)
        self.assertEqual(r(500), 17)
        self.assertGreaterEqual(r(1000), r(100))   # monotonic: bigger set → ≥ as many batches

    def test_small_run_sets_no_batch_env(self):
        import os
        saved = os.environ.pop("JUNE_BENCH_POOL_INGEST_BATCHES", None)
        try:
            reproduce._ask_ingest_batches(30)      # non-tty → returns early; also rec==1
            self.assertNotIn("JUNE_BENCH_POOL_INGEST_BATCHES", os.environ)
        finally:
            if saved is not None:
                os.environ["JUNE_BENCH_POOL_INGEST_BATCHES"] = saved


class TestPresetWiring(unittest.TestCase):
    def test_apply_env_bakes_the_preset(self):
        import os
        keys = ("JUNE_BENCH_JUNE_URL", "JUNE_BENCH_JUNE_KEY", "JUNE_BENCH_JUNE_POOL",
                "JUNE_BENCH_JUNE_BACKFILL", "JUNE_BENCH_LLM_KEY", "JUNE_BENCH_LLM_MODEL")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            os.environ.pop("JUNE_BENCH_JUNE_URL", None)
            reproduce._apply_env("acc-key", "llm-key", "anthropic/claude-opus-4-8")
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_URL"], reproduce.PRESET["url"])
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_KEY"], "acc-key")
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_POOL"], "1")
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_BACKFILL"], "1")
            self.assertEqual(os.environ["JUNE_BENCH_LLM_KEY"], "llm-key")   # BYO — caller pays
            self.assertEqual(os.environ["JUNE_BENCH_LLM_MODEL"], "anthropic/claude-opus-4-8")  # BYO model
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_preset_is_open_pool_hotpot(self):
        self.assertEqual(reproduce.PRESET["dataset"], "hotpot")
        self.assertEqual(reproduce.PRESET["split"], "full")


class TestBundledDataset(unittest.TestCase):
    def test_slice_ships_and_loads_offline(self):
        # the 100-question slice must be IN the package so reproduce never needs a download
        ds = reproduce._BundledHotpot()
        exs = ds.load(limit=None)
        self.assertEqual(len(exs), 100)
        first = exs[0]
        self.assertTrue(first.question and first.golds)
        self.assertTrue(first.context, "hotpot examples must carry their distractor passages")

    def test_limit_is_respected(self):
        self.assertEqual(len(reproduce._BundledHotpot().load(limit=5)), 5)

    def test_fixture_file_is_present_in_package(self):
        import pathlib

        from june_bench.datasets import loaders
        p = pathlib.Path(loaders._FIXTURES) / "hotpot_reproduce.json"
        self.assertTrue(p.exists(), "bundled reproduce slice missing from the package data")


class TestBundledRetrievalSlices(unittest.TestCase):
    def test_all_three_slices_ship_and_load_with_gold(self):
        for ds in ("locomo", "longmemeval", "financebench"):
            exs = reproduce._load_retrieval_examples(ds, limit=None)
            self.assertGreater(len(exs), 0, f"{ds}: no examples")
            # at least some query must have gold doc-ids present in the pooled corpus, or recall is
            # 0 by construction (the slice guarantees gold docs are bundled)
            total_gold = sum(len(e.meta.get("gold_ids", []) or []) for e in exs)
            self.assertGreater(total_gold, 0, f"{ds}: no gold_ids")
            # and each example carries id-preserving corpus docs for the ingest pool
            self.assertTrue(any(e.meta.get("corpus_docs") for e in exs), f"{ds}: no corpus_docs")

    def test_gold_docs_are_actually_in_the_pool(self):
        # the whole point of the re-slice: a query's gold must exist among the ingested docs
        exs = reproduce._load_retrieval_examples("locomo", limit=None)
        pool_ids = {i for e in exs for (i, _t) in e.meta.get("corpus_docs", [])}
        golds = {g for e in exs for g in e.meta.get("gold_ids", [])}
        self.assertTrue(golds, "no gold to check")
        self.assertTrue(golds & pool_ids, "gold docs are missing from the bundled pool (recall would be 0)")

    def test_retrieval_how_line_no_model_names(self):
        for how in (reproduce._RETRIEVAL_HOW_HAYSTACK, reproduce._RETRIEVAL_HOW_POOL):
            low = how.lower()
            for term in _BANNED:
                self.assertNotIn(term, low, f"retrieval How line leaks a model name: {term!r}")


class TestResumableCheckpoint(unittest.TestCase):
    """A dropped retrieval run resumes from its checkpoint instead of restarting."""

    def _examples(self, n):
        return [types.SimpleNamespace(qid=f"q{i}", meta={"gold_ids": [f"g{i}"]}) for i in range(n)]

    class _FakeSys:
        _pooled = False

        def __init__(self, fail_from=None):
            self.fail_from, self.calls = fail_from, []

        async def retrieve(self, ex):
            self.calls.append(ex.qid)
            if self.fail_from is not None and int(ex.qid[1:]) >= self.fail_from:
                raise ConnectionError("network lost")
            return [f"g{ex.qid[1:]}", "distractor"]

        async def aclose(self):
            pass

    def test_resume_skips_completed_and_finishes(self):
        import os
        import tempfile

        from june_bench.systems import june_retrieval as jr
        ck = tempfile.mktemp(suffix=".jsonl")
        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "2"
        exs = self._examples(6)
        try:
            # Run 1: endpoint dies from q3 → aborts; only the 3 successes are checkpointed.
            with self.assertRaises(RuntimeError):
                jr.run_retrieval(self._FakeSys(fail_from=3), exs, checkpoint_path=ck)
            self.assertEqual(sorted(jr._load_checkpoint(ck)), ["q0", "q1", "q2"])
            # Run 2: healthy → resume. Only the unfinished queries touch the network.
            s2 = self._FakeSys(fail_from=None)
            rankings, _ = jr.run_retrieval(s2, exs, checkpoint_path=ck)
            self.assertEqual(s2.calls, ["q3", "q4", "q5"])          # completed ones skipped
            self.assertEqual(sorted(rankings), [f"q{i}" for i in range(6)])
            self.assertEqual(rankings["q0"], ["g0", "distractor"])  # preserved from checkpoint
        finally:
            os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)
            if os.path.exists(ck):
                os.remove(ck)

    def test_no_checkpoint_path_is_unchanged_behavior(self):
        import asyncio  # noqa: F401

        from june_bench.systems import june_retrieval as jr
        rankings, golds = jr.run_retrieval(self._FakeSys(), self._examples(3))
        self.assertEqual(sorted(rankings), ["q0", "q1", "q2"])      # no file written, still works

    def test_qa_run_async_resumes_without_re_answering(self):
        # The QA path (runner.run_async) resumes the same way — and does NOT re-pay the answer model
        # for questions already answered on the aborted run.
        import asyncio
        import os
        import tempfile

        from june_bench.ports import Example, Prediction
        from june_bench.runner import _load_record_checkpoint, run_async

        class _FakeQA:
            name = "fake"
            _pooled = False

            def __init__(self, fail_from=None):
                self.fail_from, self.calls = fail_from, []

            async def answer(self, ex):
                self.calls.append(ex.qid)
                if self.fail_from is not None and int(ex.qid[1:]) >= self.fail_from:
                    raise ConnectionError("network lost")
                return Prediction(text=f"a{ex.qid[1:]}", meta={"calls": 1, "cost": 0.01})

        exs = [Example(qid=f"q{i}", question=f"Q{i}?", golds=(f"a{i}",)) for i in range(6)]
        ck = tempfile.mktemp(suffix=".jsonl")
        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "2"
        try:
            with self.assertRaises(RuntimeError):
                asyncio.run(run_async(_FakeQA(fail_from=3), exs, checkpoint_path=ck))
            self.assertEqual(sorted(_load_record_checkpoint(ck)), ["q0", "q1", "q2"])
            s2 = _FakeQA(fail_from=None)
            recs = asyncio.run(run_async(s2, exs, checkpoint_path=ck))
            self.assertEqual(s2.calls, ["q3", "q4", "q5"])          # completed answers not re-paid
            self.assertEqual(sorted(r.qid for r in recs), [f"q{i}" for i in range(6)])
            self.assertEqual(next(r.prediction for r in recs if r.qid == "q0"), "a0")
        finally:
            os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)
            if os.path.exists(ck):
                os.remove(ck)


class TestCustomFromFull(unittest.TestCase):
    """Custom N routes to the right source: ≤ bundled = offline slice; > bundled = the full set."""

    def _fake_full(self, n):
        return [types.SimpleNamespace(qid=f"q{i}", meta={"gold_ids": [f"g{i}"],
                                                         "corpus_docs": [(f"g{i}", "t")]}) for i in range(n)]

    def test_custom_within_bundled_uses_offline_slice(self):
        # ≤ the bundled sample → no fetch, exact slice from the offline fixture
        exs = reproduce._load_retrieval_examples("longmemeval", 5)
        self.assertEqual(len(exs), 5)

    def test_custom_above_bundled_slices_the_full_set(self):
        from unittest import mock
        with mock.patch.object(reproduce, "_load_full_retrieval", return_value=self._fake_full(500)):
            exs = reproduce._load_retrieval_examples("longmemeval", 100)
        self.assertEqual(len(exs), 100)                 # 100 real queries, not capped at 20

    def test_full_returns_the_whole_set(self):
        from unittest import mock
        with mock.patch.object(reproduce, "_load_full_retrieval", return_value=self._fake_full(500)):
            exs = reproduce._load_retrieval_examples("longmemeval", reproduce._FULL_SLICE)
        self.assertEqual(len(exs), 500)

    def test_custom_above_bundled_falls_back_when_full_unavailable(self):
        # offline / fetch failed → don't crash; honestly run the bundled sample instead
        from unittest import mock
        with mock.patch.object(reproduce, "_load_full_retrieval", return_value=None):
            exs = reproduce._load_retrieval_examples("longmemeval", 100)
        self.assertGreater(len(exs), 0)
        self.assertLessEqual(len(exs), 20)              # capped at the bundled sample, not an error

    def test_auto_fetch_triggers_once_when_missing_then_loads(self):
        # first load → missing (SystemExit) → auto-fetch → second load succeeds; only `want` streamed.
        import os
        from unittest import mock

        from june_bench.datasets import registry as reg
        state = {"load": 0, "fetch": 0, "want": None}

        def _fake_load(split, limit):  # noqa: ANN001
            state["load"] += 1
            state["want"] = limit
            if state["load"] == 1:
                raise SystemExit("not present locally")
            return self._fake_full(30)

        def _fake_fetch(names, data_dir, **kw):  # noqa: ANN001
            state["fetch"] += 1
            return [(names[0], "ok", "fetched")]

        saved = os.environ.pop("JUNE_BENCH_NO_AUTOFETCH", None)
        try:
            with mock.patch.object(reg, "get", return_value=types.SimpleNamespace(load=_fake_load)), \
                 mock.patch("june_bench.datasets.fetch.fetch", _fake_fetch):
                exs = reproduce._load_full_retrieval("longmemeval", 30)
            self.assertEqual(state["fetch"], 1)          # fetched exactly once
            self.assertEqual(state["want"], 30)          # streamed only what was asked for
            self.assertEqual(len(exs), 30)
        finally:
            if saved is not None:
                os.environ["JUNE_BENCH_NO_AUTOFETCH"] = saved

    def test_qa_ingest_retries_on_transient_5xx(self):
        # QA ingest (june_api) now rides out a transient 500 (SQLite lock) like the retrieval path.
        import asyncio
        import os
        from unittest import mock

        try:
            import httpx
        except Exception:
            self.skipTest("httpx not installed")
        from june_bench.systems.june_api import JuneApiSystem

        calls = {"n": 0}

        def _handler(req):
            calls["n"] += 1
            return httpx.Response(500) if calls["n"] < 3 else httpx.Response(200, json={"ok": True})

        os.environ["JUNE_BENCH_INGEST_RETRIES"] = "5"
        try:
            s = JuneApiSystem("http://t", transport=httpx.MockTransport(_handler))
            with mock.patch("asyncio.sleep", new=mock.AsyncMock()):   # no real backoff wait
                n = asyncio.run(s._ingest_docs(["a document"], {}))
            self.assertEqual(n, 1)               # ingested after retries
            self.assertEqual(calls["n"], 3)      # 2 × 500 then 200
        finally:
            os.environ.pop("JUNE_BENCH_INGEST_RETRIES", None)


class TestRunnerRobustness(unittest.TestCase):
    """Locks in the audit fixes: client lifecycle (C1), abort-vs-resume (H7), env hardening (H6)."""

    def _ex(self, n):
        from june_bench.ports import Example
        return [Example(qid=f"q{i}", question="?", golds=()) for i in range(n)]

    def test_runner_closes_client_on_success(self):
        import asyncio

        from june_bench.ports import Prediction
        from june_bench.runner import run_async
        closed = {"v": False}

        class _S:
            name = "s"
            _pooled = False

            async def answer(self, ex):  # noqa: ANN001
                return Prediction(text="a")

            async def aclose(self):
                closed["v"] = True

        asyncio.run(run_async(_S(), self._ex(2)))
        self.assertTrue(closed["v"])                    # C1: client released on the happy path

    def test_runner_closes_client_even_on_abort(self):
        import asyncio
        import os

        from june_bench.runner import run_async
        closed = {"v": False}

        class _S:
            name = "s"
            _pooled = False

            async def answer(self, ex):  # noqa: ANN001
                raise ConnectionError("endpoint down")

            async def aclose(self):
                closed["v"] = True

        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                asyncio.run(run_async(_S(), self._ex(3)))
        finally:
            os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)
        self.assertTrue(closed["v"])                    # C1: released via finally even when it aborts

    def test_abort_not_defeated_by_resumed_skips(self):
        # H7: a dead endpoint interspersed with resumed hits must STILL abort — a checkpoint skip must
        # not reset the consecutive-error counter (else the honesty guard never trips).
        import asyncio
        import os
        import tempfile

        from june_bench.ports import Prediction, Record
        from june_bench.runner import _append_record_checkpoint, run_async
        ck = tempfile.mktemp(suffix=".jsonl")
        _append_record_checkpoint(ck, Record(qid="q1", question="?", golds=(), prediction="done"))

        class _Dead:
            name = "s"
            _pooled = False

            async def answer(self, ex):  # noqa: ANN001
                raise ConnectionError("down")

        os.environ["JUNE_BENCH_MAX_CONSEC_ERRORS"] = "2"
        try:
            with self.assertRaises(RuntimeError):        # q0 fail(1) · q1 skip(no reset) · q2 fail(2) → abort
                asyncio.run(run_async(_Dead(), self._ex(3), checkpoint_path=ck))
        finally:
            os.environ.pop("JUNE_BENCH_MAX_CONSEC_ERRORS", None)
            if os.path.exists(ck):
                os.remove(ck)

    def test_checkpoint_lock_is_exclusive(self):
        # H3: a second run on the same checkpoint is denied (warns) so appends can't interleave.
        import os
        import tempfile
        try:
            import fcntl  # noqa: F401
        except Exception:
            self.skipTest("no fcntl (non-Unix)")
        from june_bench._util import try_lock_checkpoint
        p = tempfile.mktemp(suffix=".jsonl")
        a = try_lock_checkpoint(p)
        try:
            self.assertIsNotNone(a)                      # first run holds the lock
            self.assertIsNone(try_lock_checkpoint(p))    # concurrent run denied
        finally:
            if a is not None:
                a.close()
            for f in (p, p + ".lock"):
                if os.path.exists(f):
                    os.remove(f)

    def test_probe_config_no_url_is_empty(self):
        # H4: shared health probe — no URL → {} (not an exception), narrow fail-soft.
        from june_bench._util import probe_config
        self.assertEqual(probe_config("", "k"), {})

    def test_env_int_never_crashes_on_bad_value(self):
        import os

        from june_bench._util import env_int
        os.environ["JUNE_BENCH_TEST_BAD"] = "not-a-number"
        try:
            self.assertEqual(env_int("JUNE_BENCH_TEST_BAD", 8, lo=1, hi=100), 8)   # warns, uses default
            os.environ["JUNE_BENCH_TEST_BAD"] = "9999"
            self.assertEqual(env_int("JUNE_BENCH_TEST_BAD", 8, lo=1, hi=100), 100)  # clamped
        finally:
            os.environ.pop("JUNE_BENCH_TEST_BAD", None)


class TestInputResolution(unittest.TestCase):
    def _args(self, **kw):
        base = dict(key="", llm_key="", model="", questions=0, no_judge=False, show_config=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_flags_take_precedence_no_prompt(self):
        # flags supplied → no prompting needed even in a non-tty test env
        access, llm, model, limit, label = reproduce._resolve_inputs(
            self._args(key="K", llm_key="L", model="anthropic/claude-opus-4-8", questions=5))
        self.assertEqual((access, llm, model, limit), ("K", "L", "anthropic/claude-opus-4-8", 5))

    def test_model_defaults_to_gpt4o_non_tty(self):
        _a, _l, model, _lim, _s = reproduce._resolve_inputs(self._args(key="K", llm_key="L"))
        self.assertEqual(model, reproduce._DEFAULT_MODEL)

    def test_missing_keys_non_tty_errors_clearly(self):
        # no flags AND no env keys, non-tty (test runner) → SystemExit(2), not a silent bad run.
        # Clear the env first: a dev shell may already export these (they take precedence over prompts).
        import os
        saved = {k: os.environ.pop(k, None) for k in
                 ("JUNE_BENCH_JUNE_KEY", "JUNE_BENCH_LLM_KEY", "OPENROUTER_API_KEY")}
        try:
            with self.assertRaises(SystemExit) as cm:
                reproduce._resolve_inputs(self._args())
            self.assertEqual(cm.exception.code, 2)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_default_full_run_is_100(self):
        # flags supply the keys, so env is irrelevant here; non-tty → default full run
        _a, _l, _m, limit, label = reproduce._resolve_inputs(self._args(key="K", llm_key="L"))
        self.assertEqual(limit, 100)          # non-tty default → full headline
        self.assertEqual(label, "full headline")


class TestByoModelHeader(unittest.TestCase):
    def _mock(self):
        import httpx
        return httpx.MockTransport(lambda req: httpx.Response(200, json={}))

    def test_x_llm_model_header_sent_when_set(self):
        try:
            import httpx  # noqa: F401
        except Exception:
            self.skipTest("httpx not installed")
        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=self._mock(), llm_key="k",
                          llm_model="anthropic/claude-opus-4-8")
        self.assertEqual(s._client.headers.get("X-LLM-Model"), "anthropic/claude-opus-4-8")

    def test_no_model_header_when_unset(self):
        try:
            import httpx  # noqa: F401
        except Exception:
            self.skipTest("httpx not installed")
        from june_bench.systems.june_api import JuneApiSystem
        s = JuneApiSystem("http://t", transport=self._mock(), llm_key="k")
        self.assertIsNone(s._client.headers.get("X-LLM-Model"))


class TestCliWiring(unittest.TestCase):
    def test_reproduce_subcommand_parses(self):
        from june_bench.cli import build_parser
        args = build_parser().parse_args(["reproduce", "--key", "k", "--llm-key", "l",
                                          "--model", "anthropic/claude-opus-4-8",
                                          "--questions", "24", "--show-config"])
        self.assertEqual(args.key, "k")
        self.assertEqual(args.model, "anthropic/claude-opus-4-8")
        self.assertEqual(args.questions, 24)
        self.assertTrue(args.show_config)
        self.assertTrue(callable(args.func))


if __name__ == "__main__":
    unittest.main()
