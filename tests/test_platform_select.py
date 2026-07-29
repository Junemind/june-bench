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
