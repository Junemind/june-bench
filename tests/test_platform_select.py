"""BYO-platform plumbing (July 2026): the serving platform is part of the experiment.

Pure tests — no network. Verifies the ENUM→header contract (never a URL), the openrouter-default
omission (old endpoints stay byte-identical), and the env plumb through from_env.
"""
from __future__ import annotations

import os
import unittest

from june_bench.systems.june_api import JuneApiSystem, from_env


class TestPlatformHeader(unittest.TestCase):
    def _headers(self, **kw):
        s = JuneApiSystem("http://x", api_key="k", **kw)
        return s._client.headers

    def test_platform_header_sent_for_direct_platforms(self):
        h = self._headers(llm_key="sk", llm_platform="anthropic")
        self.assertEqual(h.get("X-LLM-Platform"), "anthropic")

    def test_openrouter_default_sends_no_header(self):
        # old endpoints predate the header — the default platform must leave requests byte-identical
        for p in ("", "openrouter", "OpenRouter"):
            h = self._headers(llm_key="sk", llm_platform=p)
            self.assertNotIn("X-LLM-Platform", h, f"platform={p!r} must not send the header")

    def test_platform_normalized_lowercase(self):
        h = self._headers(llm_key="sk", llm_platform="  OpenAI ")
        self.assertEqual(h.get("X-LLM-Platform"), "openai")


class TestPlatformEnvPlumb(unittest.TestCase):
    def setUp(self):
        self._saved = {v: os.environ.get(v) for v in
                       ("JUNE_BENCH_JUNE_URL", "JUNE_BENCH_LLM_PLATFORM")}
        os.environ["JUNE_BENCH_JUNE_URL"] = "http://x"

    def tearDown(self):
        for v, val in self._saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

    def test_from_env_reads_platform(self):
        # header is independent of the key: the platform choice must reach the endpoint even on
        # keyless probes, so validation fails fast rather than at the first paid call
        os.environ["JUNE_BENCH_LLM_PLATFORM"] = "google"
        s = from_env()
        self.assertEqual(s._client.headers.get("X-LLM-Platform"), "google")

    def test_from_env_platform_with_key(self):
        os.environ["JUNE_BENCH_LLM_PLATFORM"] = "google"
        os.environ["JUNE_BENCH_LLM_KEY"] = "sk"
        try:
            s = from_env()
            self.assertEqual(s._client.headers.get("X-LLM-Platform"), "google")
        finally:
            os.environ.pop("JUNE_BENCH_LLM_KEY", None)


if __name__ == "__main__":
    unittest.main()


# ── 2026-07-29 regression class: "written in the OpenRouter era, platform bolted on". Four shipped
# instances (key guard, judge id, h2h pinned id, auth/predates conflation) were each found by a live
# operator run. These tests pin the CLASS, not just the instances. ──

class TestPlatformNativeKeys:
    """Non-interactive runs must find the platform's NATIVE key env — not only OPENROUTER_API_KEY."""

    def test_every_menu_platform_has_key_envs(self):
        from june_bench.reproduce import _PLATFORM_KEY_ENVS, _PLATFORM_MENU
        for pid, _label, _keys in _PLATFORM_MENU.values():
            assert pid in _PLATFORM_KEY_ENVS, f"platform {pid!r} has no native key-env mapping"

    def test_native_key_env_resolves(self, monkeypatch):
        from june_bench.reproduce import _platform_env_key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k-a")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert _platform_env_key("anthropic") == "k-a"
        assert _platform_env_key("openrouter") == ""


class TestPlatformNativeModelIds:
    """Vendor-prefixed (OpenRouter-shaped) ids must never reach a direct platform verbatim."""

    def test_h2h_pinned_default_translates(self):
        from june_bench.reproduce_h2h import _DEFAULT_MODEL, _native_model_id
        assert _native_model_id(_DEFAULT_MODEL, "openai") == "gpt-4o"
        assert _native_model_id(_DEFAULT_MODEL, "openrouter") == _DEFAULT_MODEL

    def test_matrix(self):
        from june_bench.reproduce_h2h import _native_model_id
        assert _native_model_id("anthropic/claude-opus-4-8", "anthropic") == "claude-opus-4-8"
        assert _native_model_id("google/gemini-2.5-flash", "google") == "gemini-2.5-flash"
        assert _native_model_id("gpt-4o", "openai") == "gpt-4o"          # already native

    def test_reproduce_menus_are_native_on_direct_platforms(self):
        from june_bench.reproduce import _PLATFORM_MODELS
        for plat, menu in _PLATFORM_MODELS.items():
            for mid, _note in menu:
                if plat != "openrouter":
                    assert "/" not in mid, f"{plat} menu holds vendor-prefixed id {mid!r}"

    def test_judge_ids_are_native_and_wellformed(self):
        import re
        from june_bench.reproduce import _PLATFORM_JUDGE
        for plat, (_url, mid) in _PLATFORM_JUDGE.items():
            assert "/" not in mid, f"judge id for {plat} is vendor-prefixed: {mid!r}"
        # Anthropic ids are family-tier-major-minor ("claude-sonnet-4-5") — the shipped 0.1.0 judge
        # id "claude-4.5-sonnet" 404'd every judge call and printed judged-correct 0% silently.
        assert re.fullmatch(r"claude-[a-z]+-\d+(-\d+)?", _PLATFORM_JUDGE["anthropic"][1]), \
            f"anthropic judge id malformed: {_PLATFORM_JUDGE['anthropic'][1]!r}"


class TestAuthVsCapabilityConflation:
    """A rejected key must read as an AUTH problem, never as 'endpoint predates platforms'."""

    def test_probe_config_marks_auth_error(self, monkeypatch):
        import httpx
        from june_bench._util import probe_config

        def handler(request):
            return httpx.Response(401, json={"detail": "missing or invalid credentials"})

        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client",
                            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
        cfg = probe_config("https://x.example", "bad-key")
        assert cfg.get("_auth_error") is True
        assert "llm_platforms" not in cfg


class TestTinyNVerdict:
    """A 5-question smoke must not print ✗ against the n=100 baseline."""

    def test_smoke_n_gets_caveat_not_cross(self):
        from june_bench.reproduce import _verdict
        v = _verdict(0.60, 1.0, (0.72, 0.85, 0.98), 5)
        assert "smoke run" in v and "✗" not in v

    def test_full_n_still_judged(self):
        from june_bench.reproduce import _verdict
        assert _verdict(0.75, 0.97, (0.72, 0.85, 0.98), 100) == "✓ reproduced"
        assert "✗" in _verdict(0.40, 0.90, (0.72, 0.85, 0.98), 100)
