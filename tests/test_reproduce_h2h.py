"""`reproduce-h2h` — the plain-language June-vs-Cognee command. Tests the pure wiring (env baking, the
opus block, embedder discovery) with fakes: no cognee, no fastembed, no keys, no network."""
from __future__ import annotations

import os
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from june_bench import reproduce_h2h  # noqa: E402


class _EnvGuard:
    KEYS = ("JUNE_BENCH_JUNE_URL", "JUNE_BENCH_JUNE_KEY", "JUNE_BENCH_JUNE_POOL",
            "JUNE_BENCH_JUNE_BACKFILL", "JUNE_BENCH_LLM_KEY", "JUNE_BENCH_LLM_MODEL",
            "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "HUGGINGFACE_TOKENIZER",
            "LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY", "LLM_INSTRUCTOR_MODE",
            "COGNEE_SKIP_CONNECTION_TEST", "COGNEE_SEARCH_TYPE", "JUNE_JUDGE_LLM_KEY")

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestApplyEnv(unittest.TestCase):
    def test_matched_stack_baked(self):
        with _EnvGuard():
            reproduce_h2h._apply_env("acc", "llm", "openai/gpt-4o", ("ORG/some-embed-v1", 1024), cot=True)
            # June side pooled (pools BOTH sides via the shared flag)
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_POOL"], "1")
            self.assertEqual(os.environ["JUNE_BENCH_JUNE_KEY"], "acc")
            self.assertEqual(os.environ["JUNE_BENCH_LLM_MODEL"], "openai/gpt-4o")
            # Cognee side: local fastembed provider, same discovered embedder, OpenRouter-custom LLM
            self.assertEqual(os.environ["EMBEDDING_PROVIDER"], "fastembed")
            self.assertEqual(os.environ["EMBEDDING_MODEL"], "ORG/some-embed-v1")
            self.assertEqual(os.environ["EMBEDDING_DIMENSIONS"], "1024")
            self.assertEqual(os.environ["LLM_PROVIDER"], "custom")
            self.assertEqual(os.environ["LLM_MODEL"], "openrouter/openai/gpt-4o")
            self.assertEqual(os.environ["COGNEE_SEARCH_TYPE"], "GRAPH_COMPLETION_COT")

    def test_one_shot_tier(self):
        with _EnvGuard():
            reproduce_h2h._apply_env("a", "l", "openai/gpt-4o", None, cot=False)
            self.assertEqual(os.environ["COGNEE_SEARCH_TYPE"], "GRAPH_COMPLETION")

    def test_mode1_sets_a_fastembed_model_never_crashes(self):
        # REGRESSION: mode 1 (embed=None) forces provider=fastembed; it MUST also set a fastembed-supported
        # model, else Cognee falls back to its OpenAI default and dies ("not supported in TextEmbedding").
        # Patch _default_fastembed so the test needs no fastembed install.
        saved = reproduce_h2h._default_fastembed
        try:
            reproduce_h2h._default_fastembed = lambda: ("ORG/tiny-local-embed", 384)
            with _EnvGuard():
                reproduce_h2h._apply_env("a", "l", "openai/gpt-4o", None, cot=True)
                self.assertEqual(os.environ["EMBEDDING_PROVIDER"], "fastembed")
                self.assertEqual(os.environ["EMBEDDING_MODEL"], "ORG/tiny-local-embed")   # NOT the OpenAI default
                self.assertNotIn("openai", os.environ["EMBEDDING_MODEL"].lower())
        finally:
            reproduce_h2h._default_fastembed = saved

    def test_openrouter_prefix_not_doubled(self):
        with _EnvGuard():
            reproduce_h2h._apply_env("a", "l", "openrouter/openai/gpt-4o", None, cot=True)
            self.assertEqual(os.environ["LLM_MODEL"], "openrouter/openai/gpt-4o")


class TestPreflight(unittest.TestCase):
    def test_opus_is_blocked(self):
        probs = reproduce_h2h._preflight("", "", "", "anthropic/claude-opus-4-8")
        self.assertTrue(any("Opus" in p and "$90" in p for p in probs))

    def test_gpt4o_not_blocked_for_model_reason(self):
        probs = reproduce_h2h._preflight("", "", "", "openai/gpt-4o")
        self.assertFalse(any("Opus" in p for p in probs))


class TestEmbedderChoice(unittest.TestCase):
    """The embedder choice: default (June's published embedder, matched) · specify (--embedder) ·
    discover (--admin-key, fail-soft to the default)."""

    def _with_fake_fastembed(self, models):
        fake = types.ModuleType("fastembed")

        class TextEmbedding:
            @staticmethod
            def list_supported_models():
                return models
        fake.TextEmbedding = TextEmbedding
        return fake

    def _args(self, **kw):
        base = {"embedder": "", "admin_key": ""}
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_norm_embedder_name_strips_provider_and_prefix(self):
        self.assertEqual(reproduce_h2h._norm_embedder_name("http:xyz-embed-large-v1+pfx"),
                         "xyz-embed-large-v1")

    def test_match_fastembed_by_suffix(self):
        models = [{"model": "ORG/some-embed-v1", "dim": 768},
                  {"model": "ORG/xyz-embed-large-v1", "dim": 1024}]
        saved = sys.modules.get("fastembed")
        try:
            sys.modules["fastembed"] = self._with_fake_fastembed(models)
            self.assertEqual(reproduce_h2h._match_fastembed("xyz-embed-large-v1"),
                             ("ORG/xyz-embed-large-v1", 1024))
            self.assertIsNone(reproduce_h2h._match_fastembed("no-such-model"))
        finally:
            if saved is None:
                sys.modules.pop("fastembed", None)
            else:
                sys.modules["fastembed"] = saved

    def test_default_is_junes_published_embedder_noninteractive(self):
        # No flags, non-interactive → the matched default (June's embedder), never None → the same-embedder
        # reproduction out of the box with nothing to export.
        got = reproduce_h2h._choose_embedder(self._args(), "http://x")
        self.assertIsNotNone(got)
        self.assertTrue(got[0].endswith(reproduce_h2h._DEFAULT_EMBEDDER),
                        f"default embedder should be June's published id, got {got[0]!r}")

    def test_specify_flag_matches_fastembed(self):
        models = [{"model": "ORG/xyz-embed-large-v1", "dim": 1024}]
        saved = sys.modules.get("fastembed")
        try:
            sys.modules["fastembed"] = self._with_fake_fastembed(models)
            got = reproduce_h2h._choose_embedder(self._args(embedder="xyz-embed-large-v1"), "http://x")
            self.assertEqual(got, ("ORG/xyz-embed-large-v1", 1024))
        finally:
            if saved is None:
                sys.modules.pop("fastembed", None)
            else:
                sys.modules["fastembed"] = saved

    def test_discover_falls_back_to_matched_default_when_unreachable(self):
        # admin key given but the admin route is unreachable → fall back to the matched default (June's
        # published embedder), never None, never crashes.
        got = reproduce_h2h._choose_embedder(self._args(admin_key="june_admin_x"),
                                             "http://127.0.0.1:1")
        self.assertIsNotNone(got)
        self.assertTrue(got[0].endswith(reproduce_h2h._DEFAULT_EMBEDDER))


class TestEmbedderDefault(unittest.TestCase):
    """Default precedence: --embedder flag > JUNE_BENCH_EMBEDDER override > the published `_DEFAULT_EMBEDDER`
    (June's dense-lane embedder). The default IS June's embedder, so the out-of-the-box run is the
    same-embedder reproduction with nothing to export; the env var only lets an operator swap it."""

    def _choose(self, embedder="", admin_key=""):
        args = types.SimpleNamespace(embedder=embedder, admin_key=admin_key)
        return reproduce_h2h._choose_embedder(args, "http://x")   # stdout captured in tests → non-tty path

    def test_env_var_overrides_the_default(self):
        import os
        saved = os.environ.get("JUNE_BENCH_EMBEDDER")
        os.environ["JUNE_BENCH_EMBEDDER"] = "org/some-embed-v1"
        try:
            got = self._choose()
            self.assertIsNotNone(got)
            self.assertEqual(got[0], "org/some-embed-v1")   # env override wins over the built-in default
        finally:
            if saved is None:
                os.environ.pop("JUNE_BENCH_EMBEDDER", None)
            else:
                os.environ["JUNE_BENCH_EMBEDDER"] = saved

    def test_explicit_flag_beats_env_override(self):
        import os
        saved = os.environ.get("JUNE_BENCH_EMBEDDER")
        os.environ["JUNE_BENCH_EMBEDDER"] = "org/env-embed"
        try:
            self.assertEqual(self._choose(embedder="org/flag-embed")[0], "org/flag-embed")
        finally:
            if saved is None:
                os.environ.pop("JUNE_BENCH_EMBEDDER", None)
            else:
                os.environ["JUNE_BENCH_EMBEDDER"] = saved

    def test_no_flag_no_env_uses_the_published_default(self):
        import os
        saved = os.environ.pop("JUNE_BENCH_EMBEDDER", None)
        try:
            got = self._choose()                          # nothing set → June's published embedder, matched
            self.assertIsNotNone(got)
            self.assertTrue(got[0].endswith(reproduce_h2h._DEFAULT_EMBEDDER))
        finally:
            if saved is not None:
                os.environ["JUNE_BENCH_EMBEDDER"] = saved


class TestEnvOrdering(unittest.TestCase):
    """Regression guard: cognee caches its LLM/embedder config at IMPORT time, and `_preflight` imports
    cognee — so `_apply_env` MUST run before `_preflight`, or Cognee's graph build raises 'LLM API key is
    not set' even with a valid key. Locks the call order without needing cognee installed."""

    def test_apply_env_runs_before_preflight(self):
        R = reproduce_h2h
        calls: list[str] = []
        saved = (R._resolve_inputs, R._choose_embedder, R._apply_env, R._preflight, R._confirm_cost)
        try:
            R._resolve_inputs = lambda args: ("acc", "llm", True, 5)
            R._choose_embedder = lambda args, url: None
            R._apply_env = lambda *a, **k: calls.append("apply_env")
            R._preflight = lambda *a, **k: (calls.append("preflight"), [])[1]   # [] problems
            R._confirm_cost = lambda limit, cot: False                          # abort → no real run
            rc = R.run_reproduce_h2h(types.SimpleNamespace(
                key="", llm_key="", model="", questions=5, cot=False,
                embedder="", admin_key="", no_judge=True))
        finally:
            (R._resolve_inputs, R._choose_embedder, R._apply_env,
             R._preflight, R._confirm_cost) = saved
        self.assertEqual(rc, 1)                                                 # aborted at cost gate
        self.assertEqual(calls[:2], ["apply_env", "preflight"],
                         "env must be applied before cognee is imported in _preflight")


class TestCost(unittest.TestCase):
    """The cost section — Cognee's CoT + graph build cost far more than June per answer."""

    def test_estimate_scales_from_metered_per_100q(self):
        self.assertAlmostEqual(reproduce_h2h._cost_estimate("cognee", 100), 10.64, places=2)
        self.assertAlmostEqual(reproduce_h2h._cost_estimate("june-api", 100), 1.30, places=2)
        self.assertAlmostEqual(reproduce_h2h._cost_estimate("cognee", 24), 10.64 * 24 / 100, places=4)
        self.assertEqual(reproduce_h2h._cost_estimate("unknown", 100), 0.0)

    def test_cognee_is_multiple_x_june(self):
        c = reproduce_h2h._COST_PER_100Q_GPT4O
        self.assertGreater(c["cognee"] / c["june-api"], 5)   # the "eats a lot" point, quantified

    def test_openrouter_usage_is_failsoft(self):
        # no key → None; never raises. (Network path is exercised live, not in unit tests.)
        self.assertIsNone(reproduce_h2h._openrouter_usage(""))

    def test_print_cost_shows_actual_measured_and_ratio(self):
        import contextlib
        import io
        results = {"june-api": {"summary": {"em": 0.67}, "judged": 0.86, "n": 100},
                   "cognee": {"summary": {"em": 0.62}, "judged": 0.83, "n": 100}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # authoritative per-phase deltas measured this run
            reproduce_h2h._print_cost(results, "openai/gpt-4o", {"june-api": 1.28, "cognee": 10.40})
        out = buf.getvalue()
        self.assertIn("$1.28", out)                       # June's real billed cost
        self.assertIn("$10.40", out)                      # Cognee's real billed cost
        self.assertIn("OpenRouter billed this run", out)
        self.assertRegex(out, r"Cognee ≈ 8× June.*measured this run")

    def test_print_cost_june_zero_is_server_side(self):
        import contextlib
        import io
        results = {"june-api": {"summary": {"em": 0.67}, "judged": None, "n": 24},
                   "cognee": {"summary": {"em": 0.62}, "judged": None, "n": 24}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            reproduce_h2h._print_cost(results, "openai/gpt-4o", {"june-api": 0.0, "cognee": 2.5})
        self.assertIn("server-side — not billed", buf.getvalue())

    def test_print_cost_falls_back_to_estimate_when_credits_unavailable(self):
        import contextlib
        import io
        results = {"cognee": {"summary": {"em": 0.6}, "judged": None, "n": 100}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            reproduce_h2h._print_cost(results, "openai/gpt-4o", {}, cot=False)   # base tier
        out = buf.getvalue()
        self.assertIn("$10.64", out)                      # base estimate used
        self.assertIn("credits API unavailable", out)

    def test_cost_estimate_is_tier_aware(self):
        # CoT fires ~4-5 LLM rounds/question → ~2x base; the estimate + fallback ratio must reflect it
        self.assertGreater(reproduce_h2h._cognee_per_100q(True),
                           reproduce_h2h._cognee_per_100q(False) * 1.5)
        import contextlib
        import io
        results = {"cognee": {"summary": {"em": 0.6}, "judged": None, "n": 100}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            reproduce_h2h._print_cost(results, "openai/gpt-4o", {}, cot=True)    # CoT tier
        out = buf.getvalue()
        self.assertIn("20.00", out)                       # CoT reference, not the $10.64 base
        self.assertNotIn("$10.64", out)
        self.assertRegex(out, r"Cognee ≈ 1[0-9]× June")   # ~15x on CoT, not 8x

    def test_confirm_cost_estimate_reflects_cot_tier(self):
        # the pre-run gate must not undershoot: CoT estimate > base estimate for the same N
        import os
        saved = os.environ.get("JUNE_BENCH_YES")
        os.environ["JUNE_BENCH_YES"] = "1"                # auto-confirm; we only exercise the math path
        try:
            # non-interactive returns True without printing; assert the constants drive a >1.5x gap instead
            base = 100 * (reproduce_h2h._COST_PER_100Q_GPT4O["june-api"]
                          + reproduce_h2h._cognee_per_100q(False)) / 100.0
            cotc = 100 * (reproduce_h2h._COST_PER_100Q_GPT4O["june-api"]
                          + reproduce_h2h._cognee_per_100q(True)) / 100.0
            self.assertGreater(cotc, base * 1.4)
            self.assertTrue(reproduce_h2h._confirm_cost(100, cot=True))   # JUNE_BENCH_YES path
        finally:
            if saved is None:
                os.environ.pop("JUNE_BENCH_YES", None)
            else:
                os.environ["JUNE_BENCH_YES"] = saved


class TestCliRegistration(unittest.TestCase):
    def test_reproduce_h2h_is_registered(self):
        from june_bench.cli import build_parser
        p = build_parser()
        # argparse subparsers live on the _SubParsersAction; just assert parse doesn't error on the name
        ns = p.parse_args(["reproduce-h2h", "--questions", "5", "--one-shot"])
        self.assertTrue(hasattr(ns, "func"))
        self.assertEqual(ns.questions, 5)
        self.assertIs(ns.cot, False)


if __name__ == "__main__":
    unittest.main()
